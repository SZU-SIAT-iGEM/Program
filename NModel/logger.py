"""
趋化性模拟建模 —— 日志与数据记录模块
=======================================
将仿真数据以CSV格式输出，便于后续分析和绘图。

输出文件：
  {label}_timeseries.csv  — 每个记录时间点的群体统计
  {label}_radial.csv      — 每个记录时间点的径向密度分布
  {label}_zdist.csv       — 每个记录时间点的z轴分布
  summary.csv             — 所有条件的汇总对比表
"""

import os
import csv
import numpy as np
from config import (
    OUTPUT_DIR, DOMAIN_SIZE,
    SUBSTRATE_POSITIONS, SUBSTRATE_RADIUS,
    DENSITY_SHELL_INNER, DENSITY_SHELL_OUTER,
    RADIAL_BINS,
)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


class SimulationLogger:
    """单组仿真的数据记录器。"""

    def __init__(self, label):
        self.label = label
        ensure_output_dir()

        # 时间序列记录
        self.ts_rows = []

        # 径向密度记录
        self.radial_rows = []

        # z分布记录
        self.zdist_rows = []

        # 底物参数
        self.sub_center = np.array(SUBSTRATE_POSITIONS[0])
        self.shell_inner = SUBSTRATE_RADIUS + DENSITY_SHELL_INNER
        self.shell_outer = SUBSTRATE_RADIUS + DENSITY_SHELL_OUTER
        self.radial_bins = np.array(RADIAL_BINS) + SUBSTRATE_RADIUS  # 从表面算起

    def record_step(self, t, bacteria, env):
        """
        记录一个时间点的完整统计数据。

        参数：
            t: float, 当前仿真时间 (s)
            bacteria: list[Bacterium], 细菌群体
            env: Environment, 环境对象
        """
        alive = [b for b in bacteria if b.alive]
        n_alive = len(alive)
        if n_alive == 0:
            return

        positions = np.array([b.position for b in alive])
        tumble_probs = np.array([b.tumble_probability for b in alive])
        activities = np.array([b.receptor_activity for b in alive])
        methylations = np.array([b.methylation for b in alive])
        states = [b.state for b in alive]
        n_running = sum(1 for s in states if s == "run")

        # 到底物中心的距离
        dists = np.linalg.norm(positions - self.sub_center, axis=1)

        # 底物附近菌数（球壳）
        n_near = np.sum((dists >= self.shell_inner) & (dists <= self.shell_outer))

        # 每个位置的浓度
        concentrations = np.array([
            env.get_concentration(b.position) for b in alive
        ])

        total_cons = sum(b.nutrient_consumed for b in bacteria)

        # === 时间序列行 ===
        self.ts_rows.append({
            "time_s": t,
            "time_min": t / 60.0,
            "n_alive": n_alive,
            "n_running": n_running,
            "n_tumbling": n_alive - n_running,
            "n_near_substrate": int(n_near),
            "frac_near_substrate": n_near / n_alive,
            "mean_tumble_prob": float(np.mean(tumble_probs)),
            "std_tumble_prob": float(np.std(tumble_probs)),
            "mean_receptor_activity": float(np.mean(activities)),
            "mean_methylation": float(np.mean(methylations)),
            "mean_distance_um": float(np.mean(dists) * 1e6),
            "median_distance_um": float(np.median(dists) * 1e6),
            "mean_concentration": float(np.mean(concentrations)),
            "mean_x_um": float(np.mean(positions[:, 0]) * 1e6),
            "mean_y_um": float(np.mean(positions[:, 1]) * 1e6),
            "mean_z_um": float(np.mean(positions[:, 2]) * 1e6),
            "std_z_um": float(np.std(positions[:, 2]) * 1e6),
            "total_consumption_mol": total_cons,
        })

        # === 径向密度 ===
        bins = self.radial_bins
        for i in range(len(bins) - 1):
            r_inner = bins[i]
            r_outer = bins[i + 1]
            n_in_shell = int(np.sum((dists >= r_inner) & (dists < r_outer)))
            # 球壳体积
            vol = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)
            density = n_in_shell / (vol * 1e18) if vol > 0 else 0  # count per μm³ → ×1e18
            self.radial_rows.append({
                "time_s": t,
                "r_inner_um": (r_inner - SUBSTRATE_RADIUS) * 1e6,
                "r_outer_um": (r_outer - SUBSTRATE_RADIUS) * 1e6,
                "r_mid_um": ((r_inner + r_outer) / 2 - SUBSTRATE_RADIUS) * 1e6,
                "count": n_in_shell,
                "shell_volume_um3": vol * 1e18,
                "density_per_um3": density,
            })

        # === z轴分布 ===
        z_vals = positions[:, 2] * 1e6  # μm
        z_bins_edges = np.linspace(0, DOMAIN_SIZE * 1e6, 21)  # 20个bin
        hist, _ = np.histogram(z_vals, bins=z_bins_edges)
        for i in range(len(hist)):
            self.zdist_rows.append({
                "time_s": t,
                "z_low_um": z_bins_edges[i],
                "z_high_um": z_bins_edges[i + 1],
                "z_mid_um": (z_bins_edges[i] + z_bins_edges[i + 1]) / 2,
                "count": int(hist[i]),
            })

    def save(self):
        """将所有记录的数据写入CSV文件。"""
        ensure_output_dir()

        # 时间序列
        if self.ts_rows:
            path = os.path.join(OUTPUT_DIR, f"{self.label}_timeseries.csv")
            _write_csv(path, self.ts_rows)
            print(f"  日志: {path} ({len(self.ts_rows)} 行)")

        # 径向密度
        if self.radial_rows:
            path = os.path.join(OUTPUT_DIR, f"{self.label}_radial.csv")
            _write_csv(path, self.radial_rows)
            print(f"  日志: {path} ({len(self.radial_rows)} 行)")

        # z分布
        if self.zdist_rows:
            path = os.path.join(OUTPUT_DIR, f"{self.label}_zdist.csv")
            _write_csv(path, self.zdist_rows)
            print(f"  日志: {path} ({len(self.zdist_rows)} 行)")

    def get_summary(self):
        """返回该条件的汇总统计（用于最终对比表）。"""
        if not self.ts_rows:
            return {}

        # 取后半段时间的平均（忽略初始瞬态）
        half = len(self.ts_rows) // 2
        later = self.ts_rows[half:]

        n_near_vals = [r["n_near_substrate"] for r in later]
        p_tumble_vals = [r["mean_tumble_prob"] for r in later]
        dist_vals = [r["mean_distance_um"] for r in later]

        return {
            "label": self.label,
            "n_near_mean": np.mean(n_near_vals),
            "n_near_std": np.std(n_near_vals),
            "n_near_max": max(n_near_vals),
            "n_near_final": self.ts_rows[-1]["n_near_substrate"],
            "p_tumble_mean": np.mean(p_tumble_vals),
            "mean_dist_um": np.mean(dist_vals),
            "total_consumption": self.ts_rows[-1]["total_consumption_mol"],
        }


def write_summary(summaries):
    """将所有条件的汇总写入一个CSV文件。"""
    if not summaries:
        return
    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, "summary.csv")
    _write_csv(path, summaries)
    print(f"  汇总: {path}")

    # 同时打印格式化表格
    print("\n" + "=" * 100)
    print("  最终结果对比（后半段时间平均 ± 标准差）")
    print("=" * 100)
    header = (f"  {'条件':<25s} {'近底物菌(平均±std)':>22s} {'近底物菌(峰值)':>14s} "
              f"{'P_tumble':>10s} {'平均距离(μm)':>13s} {'总消耗(mol)':>14s}")
    print(header)
    print("-" * 100)
    for s in summaries:
        print(f"  {s['label']:<25s} "
              f"{s['n_near_mean']:8.1f} ± {s['n_near_std']:<6.1f}   "
              f"{s['n_near_max']:>10d}     "
              f"{s['p_tumble_mean']:>8.3f}   "
              f"{s['mean_dist_um']:>10.0f}    "
              f"{s['total_consumption']:>12.2e}")
    print("=" * 100)

    # 富集比
    print("\n  趋化富集比（相对 no_chemotaxis）:")
    for grav in ["1g", "0g"]:
        nc = [s for s in summaries if s["label"] == f"no_chemotaxis_{grav}"]
        if not nc:
            continue
        nc_avg = nc[0]["n_near_mean"]
        if nc_avg < 0.5:
            nc_avg = 0.5  # 避免除零
        for mode in ["wild_type", "enhanced"]:
            match = [s for s in summaries if s["label"] == f"{mode}_{grav}"]
            if match:
                ratio = match[0]["n_near_mean"] / nc_avg
                print(f"    {mode}_{grav}: {match[0]['n_near_mean']:.1f} / {nc_avg:.1f} = {ratio:.1f}x")
    print()


def _write_csv(path, rows):
    """将字典列表写入CSV。"""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

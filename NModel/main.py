"""
趋化性模拟建模 —— 主控模块
============================
用法：
  python main.py                                    # 全部6组
  python main.py --quick                            # 快速测试（300s, 100菌）
  python main.py --mode wild_type --gravity 0       # 单组
  python main.py --no-gif                           # 跳过GIF生成（节省时间）
"""

import os
import sys
import time
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="趋化性模拟建模")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["wild_type", "no_chemotaxis", "enhanced"])
    parser.add_argument("--gravity", type=int, default=None, choices=[0, 1])
    parser.add_argument("--no-gif", action="store_true", help="跳过GIF生成")
    return parser.parse_args()

args = parse_args()

import config
if args.quick:
    config.TOTAL_TIME = 300
    config.N_BACTERIA = 100
    config.RECORD_INTERVAL = 5.0
    config.SNAPSHOT_INTERVAL = 15.0
    config.N_STEPS = int(config.TOTAL_TIME / config.DT)
    config.RECORD_STEPS = int(config.RECORD_INTERVAL / config.DT)
    config.SNAPSHOT_STEPS = int(config.SNAPSHOT_INTERVAL / config.DT)

from environment import Environment
from signaling import SignalingPathway
from movement import MovementEngine
from metabolism import MetabolismEngine
from bacteria import (
    create_population, count_near_substrate,
    get_all_positions, get_population_stats,
)
from visualization import (
    plot_density_timeseries, plot_consumption_timeseries,
    plot_concentration_slice, plot_bacteria_distribution,
    plot_snapshot_grid, plot_dual_projection,
    plot_radial_density, plot_z_distribution,
    plot_enrichment_bar, generate_gif,
    ensure_output_dir,
)
from logger import SimulationLogger, write_summary


def run_single_simulation(chemotaxis_mode, gravity_on, label=None):
    if label is None:
        grav_str = "1g" if gravity_on else "0g"
        label = f"{chemotaxis_mode}_{grav_str}"

    print(f"\n{'='*60}")
    print(f"  运行条件: {label}")
    print(f"  趋化模式: {chemotaxis_mode}")
    print(f"  重力: {'ON (1g)' if gravity_on else 'OFF (0g)'}")
    print(f"  细菌数: {config.N_BACTERIA}")
    print(f"  仿真时间: {config.TOTAL_TIME}s ({config.TOTAL_TIME/60:.1f}min)")
    print(f"{'='*60}")

    config.GRAVITY_ON = gravity_on

    t0 = time.time()
    env = Environment()
    sig = SignalingPathway(chemotaxis_mode=chemotaxis_mode)
    mov = MovementEngine()
    mov.gravity = gravity_on
    met = MetabolismEngine()
    bacteria = create_population()

    # 日志记录器
    logger = SimulationLogger(label)

    # 数据收集
    time_points = []
    density_near_data = []
    total_consumption_data = []
    snapshots = {}

    sub_center = np.array(config.SUBSTRATE_POSITIONS[0])
    shell_inner = config.SUBSTRATE_RADIUS + config.DENSITY_SHELL_INNER
    shell_outer = config.SUBSTRATE_RADIUS + config.DENSITY_SHELL_OUTER
    dx = config.GRID_SPACING
    grid_volume = dx ** 3

    print(f"  初始化完成 ({time.time()-t0:.1f}s)")
    t_sim_start = time.time()
    n_steps = config.N_STEPS

    for step in range(n_steps):
        t = step * config.DT

        consumption_field = np.zeros(
            (config.GRID_RESOLUTION,) * 3, dtype=np.float64)

        for bac in bacteria:
            if not bac.alive:
                continue
            conc = env.get_concentration(bac.get_position())
            sig.update(bac, conc, config.DT)
            mov.step(bac, config.DT)
            consumed_mol = met.step(bac, conc, config.DT)
            if consumed_mol > 0:
                ix, iy, iz = env.position_to_grid_index(bac.get_position())
                delta_c = met.consumption_to_concentration(consumed_mol, grid_volume)
                consumption_field[ix, iy, iz] += delta_c

        env.update(config.DT, consumption_field)

        # 记录时间序列
        if step % config.RECORD_STEPS == 0:
            n_near = count_near_substrate(bacteria, sub_center, shell_inner, shell_outer)
            total_cons = sum(b.nutrient_consumed for b in bacteria)
            time_points.append(t)
            density_near_data.append(n_near)
            total_consumption_data.append(total_cons)

            # 详细日志
            logger.record_step(t, bacteria, env)

        # 快照（用于GIF）
        if step % config.SNAPSHOT_STEPS == 0:
            snapshots[t] = get_all_positions(bacteria).copy()

        # 进度
        if step > 0 and step % (n_steps // 10) == 0:
            pct = step / n_steps * 100
            elapsed = time.time() - t_sim_start
            eta = elapsed / (step / n_steps) - elapsed
            stats = get_population_stats(bacteria)
            print(f"  [{pct:5.1f}%] t={t:.0f}s | "
                  f"近底物={density_near_data[-1]:3d} | "
                  f"P_tumble={stats['mean_tumble_prob']:.3f} | "
                  f"running={stats['n_running']:3d} | "
                  f"z_mean={stats['mean_position'][2]*1e6:.0f}μm | "
                  f"ETA={eta:.0f}s")

    # 最终快照
    t_final = n_steps * config.DT
    snapshots[t_final] = get_all_positions(bacteria).copy()

    elapsed_total = time.time() - t_sim_start
    print(f"  仿真完成 ({elapsed_total:.1f}s)")

    # 保存日志
    logger.save()

    result = {
        "time": np.array(time_points),
        "density_near": np.array(density_near_data),
        "total_consumption": np.array(total_consumption_data),
        "snapshots": snapshots,
        "final_positions": get_all_positions(bacteria),
        "final_env": env,
        "final_stats": get_population_stats(bacteria),
        "label": label,
        "logger": logger,
    }
    return result


def run_all_conditions():
    conditions = [
        ("wild_type",     True),
        ("no_chemotaxis", True),
        ("enhanced",      True),
        ("wild_type",     False),
        ("no_chemotaxis", False),
        ("enhanced",      False),
    ]

    all_results = {}
    summaries = []

    for mode, gravity in conditions:
        grav_str = "1g" if gravity else "0g"
        label = f"{mode}_{grav_str}"
        result = run_single_simulation(mode, gravity, label)
        all_results[label] = result
        summaries.append(result["logger"].get_summary())

        # 单组图表
        print(f"  生成 {label} 图表...")
        plot_bacteria_distribution(
            result["final_positions"],
            filename=f"distribution_{label}.png",
            title=f"Final Distribution: {label}")

        plot_dual_projection(
            result["final_positions"],
            filename=f"dual_proj_{label}.png",
            title=f"{label} (XY + XZ)")

        if len(result["snapshots"]) > 1:
            plot_snapshot_grid(result["snapshots"],
                               filename=f"snapshots_{label}.png")

        # GIF（如果未禁用）
        if not args.no_gif and len(result["snapshots"]) >= 4:
            generate_gif(result["snapshots"], label,
                         projection='dual', fps=4)

    # ========== 多条件对比图 ==========
    print("\n生成多条件对比图表...")

    plot_density_timeseries(all_results, "density_comparison.png")
    plot_consumption_timeseries(all_results, "consumption_comparison.png")
    plot_radial_density(all_results, "radial_density.png")
    plot_z_distribution(all_results, "z_distribution.png")
    plot_enrichment_bar(summaries, "enrichment_bar.png")

    # 浓度场截面（XY和XZ两个方向）
    last_key = list(all_results.keys())[-1]
    plot_concentration_slice(all_results[last_key]["final_env"],
                             "concentration_xy.png", axis='z')
    plot_concentration_slice(all_results[last_key]["final_env"],
                             "concentration_xz.png", axis='y')

    # 汇总
    write_summary(summaries)

    return all_results


def main():
    ensure_output_dir()

    print("=" * 60)
    print("  趋化性模拟建模 —— 微重力下细菌趋化行为仿真")
    print("=" * 60)
    print(f"  域大小: {config.DOMAIN_SIZE*1e6:.0f} μm (3D)")
    print(f"  网格: {config.GRID_RESOLUTION}³ = {config.GRID_RESOLUTION**3:,} 网格点")
    print(f"  时间步: {config.DT} s")
    print(f"  总时间: {config.TOTAL_TIME} s ({config.TOTAL_TIME/60:.0f} min)")
    print(f"  细菌数: {config.N_BACTERIA}")
    print(f"  K_D: {config.K_D:.1e} M")
    print(f"  对流速度: {config.CONVECTION_SPEED*1e6:.1f} μm/s "
          f"(游速的{config.CONVECTION_SPEED/config.SWIM_SPEED*100:.0f}%)")
    print(f"  快照间隔: {config.SNAPSHOT_INTERVAL:.0f}s (用于GIF)")
    if args.quick:
        print("  [快速测试模式]")
    if args.no_gif:
        print("  [跳过GIF生成]")

    if args.mode is not None:
        gravity = bool(args.gravity) if args.gravity is not None else True
        result = run_single_simulation(args.mode, gravity)
        plot_bacteria_distribution(
            result["final_positions"],
            filename=f"distribution_{result['label']}.png",
            title=f"Final: {result['label']}")
        plot_dual_projection(
            result["final_positions"],
            filename=f"dual_proj_{result['label']}.png",
            title=result['label'])
        if not args.no_gif and len(result["snapshots"]) >= 4:
            generate_gif(result["snapshots"], result["label"],
                         projection='dual', fps=4)
    else:
        run_all_conditions()

    print(f"\n所有输出保存在 ./{config.OUTPUT_DIR}/ 目录下。")
    print("完成。")


if __name__ == "__main__":
    main()

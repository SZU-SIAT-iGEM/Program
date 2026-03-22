"""
趋化性模拟建模 —— 可视化模块（增强版）
=========================================
图表清单：
  1. density_comparison.png      — 多条件密度时间序列
  2. consumption_comparison.png  — 多条件消耗时间序列
  3. concentration_field.png     — 浓度场截面热力图
  4. distribution_{label}.png    — 最终空间分布（XY投影）
  5. snapshots_{label}.png       — 时间快照网格
  6. dual_projection_{label}.png — XY + XZ 双投影（展示三维性）
  7. radial_density.png          — 径向密度分布对比
  8. z_distribution.png          — z轴分布直方图（展示重力效应）
  9. enrichment_bar.png          — 富集比柱状图
  10. {label}_animation.gif      — 聚集过程动画
"""

import os
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

from config import (
    OUTPUT_DIR, DOMAIN_SIZE,
    SUBSTRATE_POSITIONS, SUBSTRATE_RADIUS,
    DENSITY_SHELL_INNER, DENSITY_SHELL_OUTER,
    RADIAL_BINS, N_BACTERIA,
)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 颜色方案
# ============================================================

STYLE_MAP = {
    "wild_type_1g":       {"color": "#2196F3", "ls": "-",  "label": "Wild type (1g)"},
    "no_chemotaxis_1g":   {"color": "#F44336", "ls": "-",  "label": "No chemotaxis (1g)"},
    "enhanced_1g":        {"color": "#4CAF50", "ls": "-",  "label": "Enhanced (1g)"},
    "wild_type_0g":       {"color": "#2196F3", "ls": "--", "label": "Wild type (0g)"},
    "no_chemotaxis_0g":   {"color": "#F44336", "ls": "--", "label": "No chemotaxis (0g)"},
    "enhanced_0g":        {"color": "#4CAF50", "ls": "--", "label": "Enhanced (0g)"},
}

def _get_style(name):
    return STYLE_MAP.get(name, {"color": "gray", "ls": "-", "label": name})


# ============================================================
# 1. 密度时间序列
# ============================================================

def plot_density_timeseries(results_dict, filename="density_comparison.png"):
    ensure_output_dir()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for name, data in results_dict.items():
        s = _get_style(name)
        t_min = data["time"] / 60.0
        ax.plot(t_min, data["density_near"],
                color=s["color"], ls=s["ls"], linewidth=2, label=s["label"])

    # 均匀分布期望线
    shell_vol = (4/3) * np.pi * ((SUBSTRATE_RADIUS + DENSITY_SHELL_OUTER)**3 - SUBSTRATE_RADIUS**3)
    domain_vol = DOMAIN_SIZE**3
    expected = N_BACTERIA * shell_vol / domain_vol
    ax.axhline(expected, color='gray', ls=':', linewidth=1, alpha=0.7,
               label=f'Uniform expectation ({expected:.1f})')

    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel(f"Bacteria within {DENSITY_SHELL_OUTER*1e6:.0f} μm of substrate", fontsize=12)
    ax.set_title("Chemotaxis-Driven Accumulation", fontsize=14)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  图: {filepath}")
    return filepath


# ============================================================
# 2. 消耗时间序列
# ============================================================

def plot_consumption_timeseries(results_dict, filename="consumption_comparison.png"):
    ensure_output_dir()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for name, data in results_dict.items():
        s = _get_style(name)
        t_min = data["time"] / 60.0
        ax.plot(t_min, data["total_consumption"],
                color=s["color"], ls=s["ls"], linewidth=2, label=s["label"])

    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Total glucose consumed (mol)", fontsize=12)
    ax.set_title("Cumulative Nutrient Consumption", fontsize=14)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))

    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  图: {filepath}")
    return filepath


# ============================================================
# 3. 浓度场截面
# ============================================================

def plot_concentration_slice(env, filename="concentration_field.png", axis='z', index=None):
    ensure_output_dir()
    slice_data = env.get_slice(axis=axis, index=index)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    extent = [0, DOMAIN_SIZE * 1e6, 0, DOMAIN_SIZE * 1e6]
    vmin = max(slice_data[slice_data > 0].min(), 1e-8) if np.any(slice_data > 0) else 1e-8
    vmax = slice_data.max()

    im = ax.imshow(slice_data.T, origin='lower', extent=extent,
                   cmap='YlOrRd', norm=LogNorm(vmin=vmin, vmax=vmax), aspect='equal')

    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    for pos in SUBSTRATE_POSITIONS:
        coords_2d = [pos[i] * 1e6 for i in range(3) if i != axis_idx]
        circle = plt.Circle(coords_2d, SUBSTRATE_RADIUS * 1e6,
                            fill=False, color='white', linewidth=2, linestyle='--')
        ax.add_patch(circle)

    fig.colorbar(im, ax=ax, label='Concentration (M)')
    ax_labels = {0: ('Y (μm)', 'Z (μm)'), 1: ('X (μm)', 'Z (μm)'), 2: ('X (μm)', 'Y (μm)')}
    ax.set_xlabel(ax_labels[axis_idx][0], fontsize=12)
    ax.set_ylabel(ax_labels[axis_idx][1], fontsize=12)
    ax.set_title(f"Concentration field ({axis}={index or 'mid'} slice)", fontsize=14)

    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  图: {filepath}")
    return filepath


# ============================================================
# 4. 细菌分布（单投影）
# ============================================================

def plot_bacteria_distribution(positions, filename="bacteria_distribution.png",
                                title="Bacterial Spatial Distribution", projection='xy'):
    ensure_output_dir()
    proj_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    label_map = {'xy': ('X (μm)', 'Y (μm)'), 'xz': ('X (μm)', 'Z (μm)'), 'yz': ('Y (μm)', 'Z (μm)')}
    ax1, ax2 = proj_map[projection]
    xlabel, ylabel = label_map[projection]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    if len(positions) > 0:
        ax.scatter(positions[:, ax1] * 1e6, positions[:, ax2] * 1e6,
                   s=3, alpha=0.5, c='#2196F3', edgecolors='none')

    _draw_substrate_markers(ax, ax1, ax2)
    ax.set_xlim(0, DOMAIN_SIZE * 1e6)
    ax.set_ylim(0, DOMAIN_SIZE * 1e6)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  图: {filepath}")
    return filepath


# ============================================================
# 5. 时间快照网格
# ============================================================

def plot_snapshot_grid(snapshots_dict, filename="snapshot_grid.png"):
    ensure_output_dir()
    times = sorted(snapshots_dict.keys())
    # 最多取8帧，均匀采样
    if len(times) > 8:
        indices = np.linspace(0, len(times) - 1, 8, dtype=int)
        times = [times[i] for i in indices]
    n = len(times)
    if n == 0:
        return

    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx, t in enumerate(times):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        positions = snapshots_dict[t]
        if len(positions) > 0:
            ax.scatter(positions[:, 0] * 1e6, positions[:, 1] * 1e6,
                       s=2, alpha=0.4, c='#2196F3', edgecolors='none')
        _draw_substrate_markers(ax, 0, 1, shell=False)
        ax.set_xlim(0, DOMAIN_SIZE * 1e6)
        ax.set_ylim(0, DOMAIN_SIZE * 1e6)
        ax.set_aspect('equal')
        ax.set_title(f"t = {t/60:.1f} min", fontsize=11)
        ax.grid(True, alpha=0.2)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis('off')

    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  图: {filepath}")
    return filepath


# ============================================================
# 6. 双投影图（XY + XZ 并排，展示三维性）     ★ 新增
# ============================================================

def plot_dual_projection(positions, filename="dual_projection.png", title=""):
    ensure_output_dir()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ds = DOMAIN_SIZE * 1e6
    if len(positions) > 0:
        x = positions[:, 0] * 1e6
        y = positions[:, 1] * 1e6
        z = positions[:, 2] * 1e6
        ax1.scatter(x, y, s=3, alpha=0.5, c='#2196F3', edgecolors='none')
        ax2.scatter(x, z, s=3, alpha=0.5, c='#2196F3', edgecolors='none')

    _draw_substrate_markers(ax1, 0, 1)
    _draw_substrate_markers(ax2, 0, 2)

    for ax, xlabel, ylabel, t in [
        (ax1, 'X (μm)', 'Y (μm)', 'XY projection (top view)'),
        (ax2, 'X (μm)', 'Z (μm)', 'XZ projection (side view)'),
    ]:
        ax.set_xlim(0, ds)
        ax.set_ylim(0, ds)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(t, fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    fig.suptitle(title or "3D Bacterial Distribution (dual projection)", fontsize=14, y=1.02)

    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  图: {filepath}")
    return filepath


# ============================================================
# 7. 径向密度分布对比图                        ★ 新增
# ============================================================

def plot_radial_density(results_dict, filename="radial_density.png"):
    """
    根据最终时刻的细菌位置计算径向密度，多条件对比。
    """
    ensure_output_dir()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    sub_center = np.array(SUBSTRATE_POSITIONS[0])
    bins = np.array(RADIAL_BINS) + SUBSTRATE_RADIUS
    r_mid = np.array([(bins[i] + bins[i+1]) / 2 - SUBSTRATE_RADIUS for i in range(len(bins)-1)]) * 1e6

    for name, data in results_dict.items():
        s = _get_style(name)
        positions = data["final_positions"]
        if len(positions) == 0:
            continue
        dists = np.linalg.norm(positions - sub_center, axis=1)
        densities = []
        for i in range(len(bins) - 1):
            n_in = np.sum((dists >= bins[i]) & (dists < bins[i+1]))
            vol = (4/3) * np.pi * (bins[i+1]**3 - bins[i]**3) * 1e18  # μm³
            densities.append(n_in / vol * 1e6 if vol > 0 else 0)  # per 10⁶ μm³

        ax.plot(r_mid, densities, color=s["color"], ls=s["ls"],
                linewidth=2, marker='o', markersize=4, label=s["label"])

    # 均匀分布期望
    uniform_density = N_BACTERIA / (DOMAIN_SIZE**3 * 1e18) * 1e6
    ax.axhline(uniform_density, color='gray', ls=':', linewidth=1, alpha=0.7,
               label=f'Uniform ({uniform_density:.2f})')

    ax.set_xlabel("Distance from substrate surface (μm)", fontsize=12)
    ax.set_ylabel("Density (bacteria per 10⁶ μm³)", fontsize=12)
    ax.set_title("Radial Density Profile Around Substrate", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  图: {filepath}")
    return filepath


# ============================================================
# 8. z轴分布直方图（重力效应）                  ★ 新增
# ============================================================

def plot_z_distribution(results_dict, filename="z_distribution.png"):
    """多条件z轴分布对比，体现重力沉降效应。"""
    ensure_output_dir()

    # 分开画1g和0g
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    z_bins = np.linspace(0, DOMAIN_SIZE * 1e6, 21)

    for ax, grav, title in [(ax1, "1g", "Normal gravity (1g)"), (ax2, "0g", "Microgravity (0g)")]:
        for mode in ["wild_type", "no_chemotaxis", "enhanced"]:
            name = f"{mode}_{grav}"
            if name not in results_dict:
                continue
            data = results_dict[name]
            positions = data["final_positions"]
            if len(positions) == 0:
                continue
            s = _get_style(name)
            z_vals = positions[:, 2] * 1e6
            ax.hist(z_vals, bins=z_bins, alpha=0.4, color=s["color"],
                    label=s["label"], density=True)
            # 叠加线条
            hist, _ = np.histogram(z_vals, bins=z_bins, density=True)
            z_mid = (z_bins[:-1] + z_bins[1:]) / 2
            ax.plot(z_mid, hist, color=s["color"], ls=s["ls"], linewidth=2)

        ax.set_xlabel("Z position (μm)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axvline(1000, color='orange', ls=':', linewidth=1, alpha=0.5)

    ax1.set_ylabel("Probability density", fontsize=12)
    fig.suptitle("Z-axis Distribution (gravity effect)", fontsize=14)

    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  图: {filepath}")
    return filepath


# ============================================================
# 9. 富集比柱状图                              ★ 新增
# ============================================================

def plot_enrichment_bar(summaries, filename="enrichment_bar.png"):
    """用柱状图展示各条件的时间平均富集比。"""
    ensure_output_dir()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    for ax, grav, title in [(ax1, "1g", "Normal gravity (1g)"), (ax2, "0g", "Microgravity (0g)")]:
        modes = ["no_chemotaxis", "wild_type", "enhanced"]
        labels = ["No chemo", "Wild type", "Enhanced"]
        colors = ["#F44336", "#2196F3", "#4CAF50"]
        vals = []
        errs = []
        for mode in modes:
            name = f"{mode}_{grav}"
            match = [s for s in summaries if s["label"] == name]
            if match:
                vals.append(match[0]["n_near_mean"])
                errs.append(match[0]["n_near_std"])
            else:
                vals.append(0)
                errs.append(0)

        bars = ax.bar(labels, vals, yerr=errs, capsize=5, color=colors, alpha=0.8, edgecolor='black')

        # 标注数值
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 均匀分布期望线
        shell_vol = (4/3) * np.pi * ((SUBSTRATE_RADIUS + DENSITY_SHELL_OUTER)**3 - SUBSTRATE_RADIUS**3)
        expected = N_BACTERIA * shell_vol / DOMAIN_SIZE**3
        ax.axhline(expected, color='gray', ls=':', linewidth=1.5, alpha=0.7)
        ax.text(2.4, expected + 0.5, f'Uniform\n({expected:.1f})', fontsize=9, color='gray')

        ax.set_ylabel("Mean bacteria near substrate", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_ylim(0, max(vals) * 1.4 if max(vals) > 0 else 10)

    fig.suptitle("Chemotactic Enrichment Near Substrate (time-averaged)", fontsize=14)

    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  图: {filepath}")
    return filepath


# ============================================================
# 10. GIF动画                                   ★ 新增
# ============================================================

def generate_gif(snapshots, label, filename=None, fps=4, projection='xy'):
    """
    将快照序列转化为GIF动画。

    需要 Pillow 库：pip install Pillow

    参数：
        snapshots: dict, {time_s: positions_array}
        label: str, 条件标签
        filename: str, 输出文件名
        fps: int, 帧率
        projection: str, 'xy' / 'xz' / 'dual'
    """
    ensure_output_dir()
    if filename is None:
        filename = f"{label}_animation.gif"

    try:
        from PIL import Image
    except ImportError:
        print(f"  ⚠ 生成GIF需要 Pillow 库。请运行: pip install Pillow")
        return None

    times = sorted(snapshots.keys())
    if len(times) < 2:
        return None

    frames = []
    ds = DOMAIN_SIZE * 1e6
    sub_center = np.array(SUBSTRATE_POSITIONS[0])
    shell_outer_total = SUBSTRATE_RADIUS + DENSITY_SHELL_OUTER

    for t in times:
        positions = snapshots[t]

        if projection == 'dual':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
            axes_config = [(ax1, 0, 1, 'X (μm)', 'Y (μm)'),
                           (ax2, 0, 2, 'X (μm)', 'Z (μm)')]
        else:
            fig, ax_single = plt.subplots(1, 1, figsize=(7, 7))
            p = {'xy': (0, 1, 'X (μm)', 'Y (μm)'), 'xz': (0, 2, 'X (μm)', 'Z (μm)')}[projection]
            axes_config = [(ax_single, p[0], p[1], p[2], p[3])]

        for ax, a1, a2, xl, yl in axes_config:
            if len(positions) > 0:
                # 按距底物远近着色
                dists = np.linalg.norm(positions - sub_center, axis=1)
                near = dists <= shell_outer_total
                far = ~near

                if np.any(far):
                    ax.scatter(positions[far, a1] * 1e6, positions[far, a2] * 1e6,
                               s=4, alpha=0.4, c='#90CAF9', edgecolors='none')
                if np.any(near):
                    ax.scatter(positions[near, a1] * 1e6, positions[near, a2] * 1e6,
                               s=8, alpha=0.8, c='#E53935', edgecolors='none', zorder=3)

            _draw_substrate_markers(ax, a1, a2, shell=True)
            ax.set_xlim(0, ds)
            ax.set_ylim(0, ds)
            ax.set_xlabel(xl, fontsize=10)
            ax.set_ylabel(yl, fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)

        # 计算近底物菌数
        n_near = 0
        if len(positions) > 0:
            dists = np.linalg.norm(positions - sub_center, axis=1)
            n_near = int(np.sum((dists >= SUBSTRATE_RADIUS) & (dists <= shell_outer_total)))

        fig.suptitle(f"{label}   t = {t/60:.1f} min   Near substrate: {n_near}",
                     fontsize=13, fontweight='bold')
        fig.tight_layout()

        # 渲染到内存
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frame = Image.open(buf).copy()
        frames.append(frame)
        buf.close()

    # 保存GIF
    filepath = os.path.join(OUTPUT_DIR, filename)
    if frames:
        duration = int(1000 / fps)
        frames[0].save(filepath, save_all=True, append_images=frames[1:],
                        duration=duration, loop=0)
        print(f"  GIF: {filepath} ({len(frames)} 帧, {fps} fps)")

    return filepath


# ============================================================
# 辅助函数
# ============================================================

def _draw_substrate_markers(ax, ax1, ax2, shell=True):
    """在图上标记底物颗粒和统计球壳。"""
    for pos in SUBSTRATE_POSITIONS:
        circle = plt.Circle((pos[ax1] * 1e6, pos[ax2] * 1e6),
                            SUBSTRATE_RADIUS * 1e6,
                            fill=True, facecolor='#FF9800', edgecolor='black',
                            alpha=0.7, linewidth=1.5, zorder=5)
        ax.add_patch(circle)

        if shell:
            shell_c = plt.Circle((pos[ax1] * 1e6, pos[ax2] * 1e6),
                                 (SUBSTRATE_RADIUS + DENSITY_SHELL_OUTER) * 1e6,
                                 fill=False, edgecolor='gray', linewidth=1,
                                 linestyle=':', zorder=4)
            ax.add_patch(shell_c)

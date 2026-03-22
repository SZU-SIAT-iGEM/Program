"""
趋化性模拟建模 —— 全局参数配置文件
===================================
单位规范：长度 m / 时间 s / 浓度 M / 速度 m/s / 扩散系数 m²/s
"""

import numpy as np

# ============================================================
# 仿真空间参数
# ============================================================
DOMAIN_SIZE = 2000e-6
GRID_RESOLUTION = 80
DT = 0.05
TOTAL_TIME = 1800

# ============================================================
# 细菌物理参数
# ============================================================
SWIM_SPEED = 25e-6
MEAN_RUN_TIME = 1.0
TUMBLE_ANGLE_MEAN = 62.0
TUMBLE_ANGLE_STD = 26.0
TUMBLE_DURATION = 0.1
CELL_RADIUS = 0.5e-6
SEDIMENTATION_SPEED = 0.3e-6  # Stokes定律估算

# ============================================================
# 趋化信号通路参数
# ============================================================
K_D = 5e-4                # 葡萄糖PTS趋化K_D (Mesibov & Adler 1972)
TAU_A = 0.1
ADAPTATION_TIME = 4.0
CW_BIAS_BASELINE = 0.35
HILL_COEFF = 10.3
RECEPTOR_ACTIVITY_HALF = CW_BIAS_BASELINE * (
    (1.0 - CW_BIAS_BASELINE) / CW_BIAS_BASELINE
) ** (1.0 / HILL_COEFF)
ENHANCED_KD_FACTOR = 0.2

# ============================================================
# 引诱物与扩散参数
# ============================================================
ATTRACTANT_TYPE = "glucose"
GLUCOSE_DIFFUSION_COEFF = 6.7e-10

# 对流参数 —— 仅 GRAVITY_ON=True 时生效
# v1: 1e-5 → 游速40%，人为放大重力增益  ← BUG
# v2: 1e-6 → 游速4%，微扰级，不会主导聚集行为
CONVECTION_SPEED = 1e-6

# ============================================================
# 底物颗粒参数
# ============================================================
SUBSTRATE_RADIUS = 50e-6
SUBSTRATE_POSITIONS = [(1000e-6, 1000e-6, 1000e-6)]
SUBSTRATE_SURFACE_CONC = 10e-3

# ============================================================
# 细菌群体参数
# ============================================================
N_BACTERIA = 500
INITIAL_DISTRIBUTION = "uniform"

# ============================================================
# 环境参数
# ============================================================
GRAVITY_ON = True
VISCOSITY = 1e-3
TEMPERATURE = 310

# ============================================================
# 代谢参数
# ============================================================
CONSUMPTION_RATE_MAX = 1.2e-17
K_M_GLUCOSE = 2e-6
STARVATION_THRESHOLD = 1e-7
GROWTH_ENABLED = False
DEATH_RATE = 0.0

# ============================================================
# 数据记录与输出参数
# ============================================================
RECORD_INTERVAL = 10.0
SNAPSHOT_INTERVAL = 30.0       # 每30s一帧（支持GIF动画）
DENSITY_SHELL_INNER = 0.0
DENSITY_SHELL_OUTER = 200e-6
# 径向密度统计分bin (m)
RADIAL_BINS = [0, 50e-6, 100e-6, 150e-6, 200e-6, 300e-6, 500e-6, 750e-6, 1000e-6]

OUTPUT_DIR = "results"

# ============================================================
# 派生参数（自动计算）
# ============================================================
GRID_SPACING = DOMAIN_SIZE / GRID_RESOLUTION
N_STEPS = int(TOTAL_TIME / DT)
RECORD_STEPS = int(RECORD_INTERVAL / DT)
SNAPSHOT_STEPS = int(SNAPSHOT_INTERVAL / DT)

_DIFFUSION_STABILITY = GLUCOSE_DIFFUSION_COEFF * DT / (GRID_SPACING ** 2)
assert _DIFFUSION_STABILITY < 1.0 / 6.0, (
    f"扩散方程数值不稳定！D*dt/dx²={_DIFFUSION_STABILITY:.4f}>=1/6"
)

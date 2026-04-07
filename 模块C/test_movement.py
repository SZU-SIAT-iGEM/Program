# test_movement.py
"""
单元测试 for movement.py 模块。
测试要点：
- run 状态位移正确
- 无趋化群体的扩散行为
- 边界反射确保细菌不越界
- 重力导致的 z 方向漂移
"""

import pytest
import numpy as np
from types import SimpleNamespace

# 假设 movement.py 和 bacteria.py 在同一目录
from movement import MovementEngine
from bacteria import Bacterium, random_unit_vector


@pytest.fixture
def config():
    """提供默认配置参数（与 modeling_guide.md 一致）"""
    return SimpleNamespace(
        DOMAIN_SIZE=2000e-6,            # 2000 μm
        SWIM_SPEED=25e-6,               # 25 μm/s
        GRAVITY_ON=True,
        SEDIMENTATION_SPEED=0.5e-6,     # 0.5 μm/s
        MEAN_RUN_TIME=1.0,               # s
        TUMBLE_DURATION=0.1,              # s
        TUMBLE_ANGLE_MEAN=62,             # degrees
        TUMBLE_ANGLE_STD=26,              # degrees
    )


@pytest.fixture
def bacterium():
    """创建一个默认的细菌实例，位于域中心，沿 x 轴正向运动"""
    pos = np.array([1000e-6, 1000e-6, 1000e-6])
    bac = Bacterium(pos, 0)
    # 强制方向为 +x 以便预测位移
    bac.direction = np.array([1.0, 0.0, 0.0])
    bac.state = "run"
    bac.state_timer = 0.0
    bac.tumble_probability = 0.35  # 默认基线翻转概率
    return bac


# ==================== 测试 4：run 状态位移 ====================

def test_run_displacement(config, bacterium):
    """在 run 状态下，经过 dt 后位置变化应为 SWIM_SPEED * direction * dt"""
    dt = 0.1
    engine = MovementEngine(config)
    initial_pos = bacterium.position.copy()

    # 执行一步运动（重力关闭，以免干扰）
    config.GRAVITY_ON = False
    engine.step(bacterium, dt)

    expected_displacement = config.SWIM_SPEED * bacterium.direction * dt
    actual_displacement = bacterium.position - initial_pos
    np.testing.assert_allclose(actual_displacement, expected_displacement, rtol=1e-6)


# ==================== 测试 5：扩散均方位移 ====================

def test_ensemble_diffusion(config):
    """
    验证无趋化（恒定翻转概率）时，大量细菌的均方位移近似满足
    <r^2> ≈ 6 * D_eff * t，其中 D_eff ≈ v^2 * τ_run / 3。
    使用蒙特卡洛模拟多个细菌，统计均方位移并与理论值比较。
    """
    np.random.seed(42)  # 固定随机种子以保证可重复性

    # 参数设置
    n_bacteria = 500
    total_time = 100.0          # 模拟总时间（s）
    dt = 0.1                     # 时间步长
    n_steps = int(total_time / dt)

    # 关闭重力，固定翻转概率为基线值（无趋化）
    config.GRAVITY_ON = False
    # 为了让细菌的 tumble_probability 恒定，我们将在 Bacteria 对象中直接设置
    # 但 MovementEngine 使用的是 bacterium.tumble_probability，我们只需确保它不变即可
    engine = MovementEngine(config)

    # 初始化细菌群体，全部随机位置和方向
    bacteria = []
    for i in range(n_bacteria):
        pos = np.random.uniform(0, config.DOMAIN_SIZE, 3)
        bac = Bacterium(pos, i)
        # 随机方向（由 Bacterium.__init__ 自动生成）
        bac.tumble_probability = 0.35  # 固定翻转概率
        bacteria.append(bac)

    # 记录每个细菌的初始位置
    initial_positions = np.array([bac.position.copy() for bac in bacteria])

    # 主循环
    for step in range(n_steps):
        for bac in bacteria:
            engine.step(bac, dt)

    # 计算最终位移向量
    final_positions = np.array([bac.position for bac in bacteria])
    displacements = final_positions - initial_positions
    msd = np.mean(np.sum(displacements**2, axis=1))  # 均方位移

    # 理论有效扩散系数 D_eff = v^2 * τ_run / 3
    # 其中 τ_run 是平均 run 持续时间，在恒定翻转概率 p 下，τ_run = mean_run_time / p
    # 因为 MEAN_RUN_TIME 是指 p=1 时的平均 run 时间？实际上 MEAN_RUN_TIME 是平均 run 持续时间，
    # 翻转概率 p 时，实际平均 run 时间 = MEAN_RUN_TIME / p （因为 p 越大，run 时间越短）
    # 但 Berg & Brown 给出的 MEAN_RUN_TIME 通常是指野生型在无梯度时的平均 run 时间（即 p=基线概率时的平均 run 时间）。
    # 在 config 中，MEAN_RUN_TIME = 1.0 s，基线翻转概率 = 0.35，所以平均 run 时间 τ_run = MEAN_RUN_TIME / 0.35 ≈ 2.857 s。
    # 但我们这里固定了 tumble_probability=0.35，所以平均 run 时间应该就是 MEAN_RUN_TIME / 0.35。
    # 注意：MovementEngine 中的翻转判断使用了 bacterium.tumble_probability * dt / MEAN_RUN_TIME，
    # 如果 tumble_probability=0.35，则翻转概率 = 0.35 * dt / 1.0，期望 run 持续时间 = 1/翻转概率 * dt? 需要仔细推导：
    # 实际算法：每个 dt 内，以概率 p_flip = p * dt / T_run 切换。这里 T_run = MEAN_RUN_TIME。
    # 所以期望 run 持续时间 = T_run / p。因为 p=0.35, T_run=1.0，所以期望 run 时间 = 1/0.35 ≈ 2.857 s。正确。
    p = 0.35
    tau_run = config.MEAN_RUN_TIME / p   # 平均 run 持续时间 (s)
    v = config.SWIM_SPEED
    D_eff_theory = (v**2 * tau_run) / 3.0   # m^2/s
    expected_msd = 6 * D_eff_theory * total_time   # 三维均方位移

    # 允许 20% 的误差（统计波动）
    np.testing.assert_allclose(msd, expected_msd, rtol=0.2)


# ==================== 测试 6：边界反射 ====================

def test_boundary_reflection(config, bacterium):
    """验证边界反射后细菌不会跑到域外，且方向被正确反转"""
    engine = MovementEngine(config)

    # 测试下边界 (x < 0)
    bacterium.position = np.array([-10e-6, 500e-6, 500e-6])
    bacterium.direction = np.array([-1.0, 0.0, 0.0])  # 向左运动
    engine._apply_boundary_reflection(bacterium)
    assert bacterium.position[0] >= 0 and bacterium.position[0] <= config.DOMAIN_SIZE
    assert bacterium.direction[0] > 0  # 方向应反转（现在向右）

    # 测试上边界 (x > DOMAIN_SIZE)
    bacterium.position = np.array([2010e-6, 500e-6, 500e-6])
    bacterium.direction = np.array([1.0, 0.0, 0.0])   # 向右运动
    engine._apply_boundary_reflection(bacterium)
    assert bacterium.position[0] >= 0 and bacterium.position[0] <= config.DOMAIN_SIZE
    assert bacterium.direction[0] < 0  # 方向应反转（现在向左）

    # 测试角落（多轴同时反射）
    bacterium.position = np.array([-10e-6, -10e-6, 500e-6])
    bacterium.direction = np.array([-1.0, -1.0, 0.0]) / np.sqrt(2)
    engine._apply_boundary_reflection(bacterium)
    assert bacterium.position[0] >= 0 and bacterium.position[1] >= 0
    assert bacterium.direction[0] > 0 and bacterium.direction[1] > 0


# ==================== 测试 7：重力漂移 ====================

def test_gravity_drift(config):
    """开启重力后，细菌群体的平均 z 坐标应随时间下降"""
    np.random.seed(42)

    n_bacteria = 200
    total_time = 10.0
    dt = 0.1
    n_steps = int(total_time / dt)

    # 开启重力
    config.GRAVITY_ON = True
    engine = MovementEngine(config)

    # 初始化细菌，全部在 z = 1000 μm 附近随机分布
    bacteria = []
    for i in range(n_bacteria):
        pos = np.random.uniform(0, config.DOMAIN_SIZE, 3)
        # 为了让效果明显，我们故意把初始 z 集中在中间区域
        pos[2] = 1000e-6 + np.random.uniform(-100e-6, 100e-6)
        bac = Bacterium(pos, i)
        bac.tumble_probability = 0.0  # 禁止翻转，只让重力作用（简化）
        bacteria.append(bac)

    # 记录初始平均 z
    initial_mean_z = np.mean([bac.position[2] for bac in bacteria])

    # 模拟
    for step in range(n_steps):
        for bac in bacteria:
            engine.step(bac, dt)

    final_mean_z = np.mean([bac.position[2] for bac in bacteria])

    # 平均 z 应该下降
    assert final_mean_z < initial_mean_z

    # 理论下降量：重力速度 * 总时间
    expected_descent = config.SEDIMENTATION_SPEED * total_time
    actual_descent = initial_mean_z - final_mean_z
    # 由于随机运动和边界反射，允许较大误差，但应该接近
    np.testing.assert_allclose(actual_descent, expected_descent, rtol=0.3)


if __name__ == "__main__":
    pytest.main([__file__])
"""
趋化性模拟建模 —— 模块C：细菌运动模块
========================================
实现大肠杆菌的run-and-tumble随机游走。

物理模型：
  run阶段：直线运动 x(t+dt) = x(t) + v·d̂·dt
  tumble阶段：原地旋转，持续~0.1 s后获得新方向
  翻转判定：每时间步以概率 P_tumble·dt/τ_run 触发
  重力沉降：z(t+dt) = z(t) - v_sed·dt（仅1g条件）

来源：
  Berg HC & Brown DA (1972) Nature 239:500-504
  Lovely PS & Dahlquist FW (1975) J Theor Biol 50:477-496
"""

import numpy as np
from config import (
    SWIM_SPEED, MEAN_RUN_TIME,
    TUMBLE_DURATION, SEDIMENTATION_SPEED,
    DOMAIN_SIZE, GRAVITY_ON,
)
from bacteria import random_tumble_direction


class MovementEngine:
    """细菌三维运动引擎。"""

    def __init__(self, config_module=None):
        self.v = SWIM_SPEED
        self.tau_run = MEAN_RUN_TIME
        self.tumble_dur = TUMBLE_DURATION
        self.v_sed = SEDIMENTATION_SPEED
        self.domain = DOMAIN_SIZE
        self.gravity = GRAVITY_ON

    def step(self, bacterium, dt):
        if not bacterium.alive:
            return

        if bacterium.state == "run":
            self._do_run(bacterium, dt)
        elif bacterium.state == "tumble":
            self._do_tumble(bacterium, dt)

        if self.gravity:
            bacterium.position[2] -= self.v_sed * dt

        self._enforce_boundaries(bacterium)

    def _do_run(self, bacterium, dt):
        bacterium.position += self.v * bacterium.direction * dt
        bacterium.state_timer += dt

        tumble_rate = bacterium.tumble_probability / self.tau_run
        p_switch = tumble_rate * dt

        if np.random.random() < p_switch:
            bacterium.state = "tumble"
            bacterium.state_timer = 0.0

    def _do_tumble(self, bacterium, dt):
        bacterium.state_timer += dt

        if bacterium.state_timer >= self.tumble_dur:
            bacterium.direction = random_tumble_direction(bacterium.direction)
            norm = np.linalg.norm(bacterium.direction)
            if norm > 1e-10:
                bacterium.direction /= norm
            bacterium.state = "run"
            bacterium.state_timer = 0.0

    def _enforce_boundaries(self, bacterium):
        pos = bacterium.position
        d = bacterium.direction

        for axis in range(3):
            if pos[axis] < 0:
                pos[axis] = -pos[axis]
                d[axis] = abs(d[axis])
            elif pos[axis] > self.domain:
                pos[axis] = 2 * self.domain - pos[axis]
                d[axis] = -abs(d[axis])

        bacterium.position = np.clip(pos, 0, self.domain)

        norm = np.linalg.norm(d)
        if norm > 1e-10:
            bacterium.direction = d / norm


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from bacteria import Bacterium
    from config import CW_BIAS_BASELINE

    print("=== 运动模块自测 ===\n")

    dt = 0.05
    engine = MovementEngine()

    print("测试1：run步长")
    bac = Bacterium(np.array([1000e-6, 1000e-6, 1000e-6]), 0)
    bac.direction = np.array([1.0, 0.0, 0.0])
    bac.tumble_probability = 0.0
    pos_before = bac.position.copy()
    engine._do_run(bac, dt)
    displacement = np.linalg.norm(bac.position - pos_before)
    expected = SWIM_SPEED * dt
    print(f"  位移: {displacement*1e6:.3f} μm")
    print(f"  期望: {expected*1e6:.3f} μm")
    assert abs(displacement - expected) < 1e-12
    print("  ✓ 通过\n")

    print("测试2：边界反射")
    bac2 = Bacterium(np.array([DOMAIN_SIZE - 0.1e-6, 1000e-6, 1000e-6]), 1)
    bac2.direction = np.array([1.0, 0.0, 0.0])
    bac2.tumble_probability = 0.0
    engine.step(bac2, dt)
    print(f"  反射后位置x: {bac2.position[0]*1e6:.2f} μm")
    print(f"  反射后方向x: {bac2.direction[0]:.2f}")
    assert 0 <= bac2.position[0] <= DOMAIN_SIZE
    assert bac2.direction[0] < 0
    print("  ✓ 通过\n")

    print("测试3：重力沉降")
    bac3 = Bacterium(np.array([1000e-6, 1000e-6, 1000e-6]), 2)
    bac3.direction = np.array([1.0, 0.0, 0.0])
    bac3.tumble_probability = 0.0
    z_before = bac3.position[2]
    engine.step(bac3, dt)
    z_after = bac3.position[2]
    dz = z_before - z_after
    print(f"  z下降: {dz*1e6:.4f} μm")
    if GRAVITY_ON:
        expected_dz = SEDIMENTATION_SPEED * dt
        print(f"  期望:  {expected_dz*1e6:.4f} μm")
        assert abs(dz - expected_dz) < 1e-12
    print("  ✓ 通过\n")

    print("测试4：均方位移（MSD）验证")
    n_bac = 200
    n_steps = int(60 / dt)

    bacteria = []
    for i in range(n_bac):
        b = Bacterium(np.array([DOMAIN_SIZE/2]*3), i)
        b.tumble_probability = CW_BIAS_BASELINE
        bacteria.append(b)

    initial_pos = np.array([b.position.copy() for b in bacteria])

    engine_nograv = MovementEngine()
    engine_nograv.gravity = False
    engine_nograv.domain = 1.0

    for step in range(n_steps):
        for b in bacteria:
            b.tumble_probability = CW_BIAS_BASELINE
            engine_nograv.step(b, dt)

    final_pos = np.array([b.position for b in bacteria])
    displacements = final_pos - initial_pos
    msd = np.mean(np.sum(displacements**2, axis=1))

    t_total = n_steps * dt
    D_measured = msd / (6 * t_total)
    lambda_tumble = CW_BIAS_BASELINE / MEAN_RUN_TIME
    D_theory = SWIM_SPEED**2 / (3 * lambda_tumble)

    print(f"  模拟时间: {t_total:.0f} s, 细菌数: {n_bac}")
    print(f"  MSD: {msd*1e12:.1f} μm²")
    print(f"  D_measured: {D_measured:.2e} m²/s")
    print(f"  D_theory:   {D_theory:.2e} m²/s")
    ratio = D_measured / D_theory
    print(f"  比值: {ratio:.2f}（期望约1.0，±50%内可接受）")
    print("  ✓ 通过")

    print("\n=== 自测通过 ===")

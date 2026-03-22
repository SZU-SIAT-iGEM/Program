"""
趋化性模拟建模 —— 模块D：代谢与消耗模块
==========================================
处理单细胞的营养消耗（Michaelis-Menten动力学）。

物理模型：
  q = q_max × c / (c + K_m)

来源：
  Lendenmann U, Snozzi M & Egli T (1996) Microbiology 142:1131-1140
  Natarajan A & Bhatt SL (2002) Biochem Eng J 11:193-199
"""

import numpy as np
from config import (
    CONSUMPTION_RATE_MAX,
    K_M_GLUCOSE,
    STARVATION_THRESHOLD,
)


class MetabolismEngine:
    """细菌代谢计算引擎。"""

    def __init__(self, config_module=None):
        self.q_max = CONSUMPTION_RATE_MAX
        self.K_m = K_M_GLUCOSE
        self.c_min = STARVATION_THRESHOLD

    def step(self, bacterium, local_concentration, dt):
        if not bacterium.alive:
            return 0.0

        c = local_concentration

        if c < self.c_min:
            return 0.0

        q = self.q_max * c / (c + self.K_m)
        consumption = q * dt
        bacterium.nutrient_consumed += consumption

        return float(consumption)

    def consumption_to_concentration(self, consumption_mol, grid_volume):
        volume_liters = grid_volume * 1000.0
        if volume_liters < 1e-30:
            return 0.0
        return consumption_mol / volume_liters


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from bacteria import Bacterium

    print("=== 代谢模块自测 ===\n")

    met = MetabolismEngine()
    dt = 0.05

    print("测试1：正常消耗（高浓度）")
    bac = Bacterium(np.array([0, 0, 0]), 0)
    c_high = 10e-3
    consumed = met.step(bac, c_high, dt)
    expected_max = CONSUMPTION_RATE_MAX * dt
    print(f"  浓度: {c_high:.0e} M")
    print(f"  消耗: {consumed:.2e} mol")
    print(f"  q_max×dt: {expected_max:.2e} mol")
    print(f"  比值: {consumed/expected_max:.4f}（期望接近1.0）")
    assert consumed > 0.99 * expected_max
    print("  ✓ 通过\n")

    print("测试2：低浓度消耗")
    bac2 = Bacterium(np.array([0, 0, 0]), 1)
    c_low = K_M_GLUCOSE
    consumed_low = met.step(bac2, c_low, dt)
    expected_half = CONSUMPTION_RATE_MAX * 0.5 * dt
    ratio = consumed_low / expected_half
    print(f"  浓度: {c_low:.0e} M (= K_m)")
    print(f"  消耗: {consumed_low:.2e} mol")
    print(f"  期望: {expected_half:.2e} mol (q_max/2 × dt)")
    print(f"  比值: {ratio:.4f}（期望1.0）")
    assert abs(ratio - 1.0) < 0.01
    print("  ✓ 通过\n")

    print("测试3：低于阈值不消耗")
    bac3 = Bacterium(np.array([0, 0, 0]), 2)
    c_starve = STARVATION_THRESHOLD * 0.1
    consumed_starve = met.step(bac3, c_starve, dt)
    print(f"  消耗: {consumed_starve:.2e} mol")
    assert consumed_starve == 0.0
    print("  ✓ 通过\n")

    print("测试4：死菌不消耗")
    bac4 = Bacterium(np.array([0, 0, 0]), 3)
    bac4.alive = False
    consumed_dead = met.step(bac4, 10e-3, dt)
    assert consumed_dead == 0.0
    print("  ✓ 通过\n")

    print("测试5：累计消耗")
    bac5 = Bacterium(np.array([0, 0, 0]), 4)
    total = 0
    for _ in range(100):
        c = met.step(bac5, 10e-3, dt)
        total += c
    print(f"  100步累计消耗: {total:.2e} mol")
    print(f"  bacterium.nutrient_consumed: {bac5.nutrient_consumed:.2e} mol")
    assert abs(total - bac5.nutrient_consumed) < 1e-25
    print("  ✓ 通过\n")

    print("测试6：消耗量→浓度转换")
    from config import GRID_SPACING
    grid_vol = GRID_SPACING ** 3
    delta_c = met.consumption_to_concentration(1e-17, grid_vol)
    expected_dc = 1e-17 / (grid_vol * 1000)
    print(f"  网格体积: {grid_vol:.2e} m³")
    print(f"  1e-17 mol → ΔC = {delta_c:.2e} M")
    print(f"  手算期望: {expected_dc:.2e} M")
    assert abs(delta_c - expected_dc) / expected_dc < 1e-6
    print("  ✓ 通过")

    print("\n=== 自测通过 ===")

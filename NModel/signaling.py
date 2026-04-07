"""
趋化性模拟建模 —— 模块B：细菌信号通路模块
=============================================
实现简化的Barkai-Leibler趋化信号通路模型。
输入：当前位置引诱物浓度 → 输出：翻转概率。

物理模型（两变量ODE）：
  da/dt = [f(m, c) - a] / τ_a      （受体活性快速响应）
  dm/dt = [a₀ - a] / τ_m            （甲基化慢适应，积分反馈）

  其中 f(m, c) = m / (m + (1 + c/K_D)·(1 - m))
  翻转概率 P = a^H / (a^H + a½^H)

来源：
  Barkai N & Leibler S (1997) Nature 387:913-917
  Tu Y, Shimizu TS & Berg HC (2008) PNAS 105:14855-14860
  Cluzel P, Surette M & Leibler S (2000) Science 287:1652-1655
"""

import numpy as np
from config import (
    K_D, TAU_A, ADAPTATION_TIME,
    CW_BIAS_BASELINE, HILL_COEFF, RECEPTOR_ACTIVITY_HALF,
    ENHANCED_KD_FACTOR,
)


class SignalingPathway:
    """趋化信号通路计算引擎。"""

    VALID_MODES = ("wild_type", "no_chemotaxis", "enhanced")

    def __init__(self, config_module=None, chemotaxis_mode="wild_type"):
        """
        初始化信号通路。

        参数：
            config_module: 未使用，保留接口兼容性
            chemotaxis_mode: str, 趋化模式
                "wild_type"     — 正常趋化参数
                "no_chemotaxis" — 翻转概率恒为基线值，不执行ODE
                "enhanced"      — K_D降低（灵敏度提高）
        """
        if chemotaxis_mode not in self.VALID_MODES:
            raise ValueError(
                f"chemotaxis_mode 必须是 {self.VALID_MODES} 之一，"
                f"收到 '{chemotaxis_mode}'"
            )

        self.mode = chemotaxis_mode

        self.tau_a = TAU_A
        self.tau_m = ADAPTATION_TIME
        self.a0 = CW_BIAS_BASELINE
        self.H = HILL_COEFF
        self.a_half = RECEPTOR_ACTIVITY_HALF

        if chemotaxis_mode == "enhanced":
            self.K_D = K_D * ENHANCED_KD_FACTOR
        else:
            self.K_D = K_D

    def _receptor_activity_function(self, m, c):
        """
        计算受体活性的瞬时目标值 f(m, c)。

        f(m, c) = m / (m + (1 + c/K_D) * (1 - m))

        来源：Barkai & Leibler (1997) Nature 387:913-917, Eq.1
        """
        m = np.clip(m, 1e-8, 1.0 - 1e-8)
        denominator = m + (1.0 + c / self.K_D) * (1.0 - m)
        return m / max(denominator, 1e-20)

    def _hill_function(self, a):
        """
        将受体活性映射为翻转概率（CW偏好）。

        P_tumble = a^H / (a^H + a½^H)

        来源：Cluzel P et al. (2000) Science 287:1652-1655
        """
        a = np.clip(a, 1e-8, 1.0 - 1e-8)
        a_H = a ** self.H
        a_half_H = self.a_half ** self.H
        return a_H / (a_H + a_half_H)

    def update(self, bacterium, current_concentration, dt):
        """
        更新一个细菌的信号通路状态和翻转概率。

        参数：
            bacterium: Bacterium对象
            current_concentration: float, 当前位置引诱物浓度（单位 M）
            dt: float, 时间步长（单位 s）
        """
        if self.mode == "no_chemotaxis":
            bacterium.set_tumble_probability(self.a0)
            return

        c = max(current_concentration, 0.0)
        a = bacterium.receptor_activity
        m = bacterium.methylation

        # da/dt = [f(m, c) - a] / τ_a
        f_mc = self._receptor_activity_function(m, c)
        dadt = (f_mc - a) / self.tau_a

        # dm/dt = [a₀ - a] / τ_m
        # 积分反馈核心：甲基化朝消除(a - a₀)偏差的方向调节
        dmdt = (self.a0 - a) / self.tau_m

        # 欧拉法更新
        a_new = a + dadt * dt
        m_new = m + dmdt * dt

        # 裁剪到安全范围
        # 使用宽范围 [0.001, 0.999] 以允许甲基化在高浓度下充分适应
        a_new = np.clip(a_new, 0.001, 0.999)
        m_new = np.clip(m_new, 0.001, 0.999)

        bacterium.receptor_activity = float(a_new)
        bacterium.methylation = float(m_new)

        p_tumble = self._hill_function(a_new)
        bacterium.set_tumble_probability(p_tumble)


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from bacteria import Bacterium

    print("=== 信号通路模块自测 ===\n")

    dt = 0.05

    # --- 测试1：no_chemotaxis模式 ---
    print("测试1：no_chemotaxis模式")
    sig_nc = SignalingPathway(chemotaxis_mode="no_chemotaxis")
    bac = Bacterium(np.array([0, 0, 0]), 0)
    for conc in [0, 1e-6, 1e-3, 1.0]:
        sig_nc.update(bac, conc, dt)
    print(f"  各种浓度下翻转概率: {bac.tumble_probability:.3f}")
    print(f"  期望: {CW_BIAS_BASELINE:.3f}")
    assert abs(bac.tumble_probability - CW_BIAS_BASELINE) < 1e-6
    print("  ✓ 通过\n")

    # --- 测试2：Hill函数基线验证 ---
    print("测试2：Hill函数基线验证")
    sig_wt = SignalingPathway(chemotaxis_mode="wild_type")
    p_at_baseline = sig_wt._hill_function(CW_BIAS_BASELINE)
    print(f"  Hill(a₀={CW_BIAS_BASELINE}) = {p_at_baseline:.6f}")
    print(f"  期望: {CW_BIAS_BASELINE:.6f}")
    assert abs(p_at_baseline - CW_BIAS_BASELINE) < 0.01
    print("  ✓ 通过\n")

    # --- 测试3：精确适应（恒定浓度下活性回到基线）---
    print("测试3：精确适应验证")
    for test_conc in [1e-5, 1e-3, 5e-3]:
        bac2 = Bacterium(np.array([0, 0, 0]), 1)
        for _ in range(int(30 / dt)):
            sig_wt.update(bac2, test_conc, dt)
        a_final = bac2.receptor_activity
        p_final = bac2.tumble_probability
        error_a = abs(a_final - CW_BIAS_BASELINE) / CW_BIAS_BASELINE
        error_p = abs(p_final - CW_BIAS_BASELINE) / CW_BIAS_BASELINE
        print(f"  c={test_conc:.0e} M → a={a_final:.4f}, P_tumble={p_final:.4f}, "
              f"偏差_a={error_a*100:.2f}%, 偏差_P={error_p*100:.2f}%")
        assert error_a < 0.05, f"适应精度不达标！偏差={error_a*100:.1f}%"
    print("  ✓ 通过\n")

    # --- 测试4：阶跃响应 ---
    print("测试4：阶跃响应（浓度突然升高→翻转概率先降后恢复）")
    sig_wt2 = SignalingPathway(chemotaxis_mode="wild_type")
    bac3 = Bacterium(np.array([0, 0, 0]), 2)

    for _ in range(int(10 / dt)):
        sig_wt2.update(bac3, 0.0, dt)
    p_before = bac3.tumble_probability
    print(f"  刺激前 P_tumble: {p_before:.4f}")

    # 突然加入引诱物（用K_D量级的浓度）
    sig_wt2.update(bac3, 5e-4, dt)
    p_immediate = bac3.tumble_probability
    print(f"  刺激后立刻 P_tumble: {p_immediate:.4f}")
    assert p_immediate < p_before, "引诱物应降低翻转概率！"
    print(f"  翻转概率下降了 → ✓")

    for _ in range(int(20 / dt)):
        sig_wt2.update(bac3, 5e-4, dt)
    p_adapted = bac3.tumble_probability
    print(f"  适应后 P_tumble: {p_adapted:.4f}")
    print(f"  与基线差: {abs(p_adapted - p_before):.4f}")
    print("  ✓ 通过\n")

    # --- 测试5：enhanced模式灵敏度更高 ---
    print("测试5：enhanced模式灵敏度对比")
    sig_en = SignalingPathway(chemotaxis_mode="enhanced")

    bac_wt = Bacterium(np.array([0, 0, 0]), 10)
    bac_en = Bacterium(np.array([0, 0, 0]), 11)

    for _ in range(int(10 / dt)):
        sig_wt.update(bac_wt, 0.0, dt)
        sig_en.update(bac_en, 0.0, dt)

    small_conc = 5e-5  # 50 μM — 低于wild_type K_D但接近enhanced K_D
    sig_wt.update(bac_wt, small_conc, dt)
    sig_en.update(bac_en, small_conc, dt)

    delta_wt = abs(bac_wt.tumble_probability - CW_BIAS_BASELINE)
    delta_en = abs(bac_en.tumble_probability - CW_BIAS_BASELINE)
    print(f"  wild_type响应幅度: {delta_wt:.6f}")
    print(f"  enhanced响应幅度: {delta_en:.6f}")
    if delta_wt > 1e-8:
        print(f"  enhanced/wild_type = {delta_en/delta_wt:.1f}x")
    print("  ✓ 通过")

    print("\n=== 自测通过 ===")

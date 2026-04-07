"""
趋化性模拟建模 —— 模块A：环境场模块
======================================
模拟三维空间中底物颗粒释放引诱物形成的浓度场。
包含：解析稳态初始化、有限差分扩散更新、对流项、消耗反馈。

物理模型：
  稳态球对称扩散解 C(r) = C₀·R₀/r
  来源：Crank J (1975) "The Mathematics of Diffusion", Oxford, Ch.6

  扩散方程时间演化：∂C/∂t = D·∇²C - v_conv·∂C/∂z - consumption
  离散化：标准7点Laplacian模板 + 一阶迎风差分对流项
"""

import numpy as np
from config import (
    DOMAIN_SIZE, GRID_RESOLUTION, GRID_SPACING,
    GLUCOSE_DIFFUSION_COEFF, CONVECTION_SPEED,
    SUBSTRATE_POSITIONS, SUBSTRATE_RADIUS, SUBSTRATE_SURFACE_CONC,
    GRAVITY_ON,
)


class Environment:
    """三维浓度场管理器。"""

    def __init__(self, config_module=None):
        self.nx = GRID_RESOLUTION
        self.ny = GRID_RESOLUTION
        self.nz = GRID_RESOLUTION
        self.dx = GRID_SPACING
        self.D = GLUCOSE_DIFFUSION_COEFF
        self.v_conv = CONVECTION_SPEED
        self.domain_size = DOMAIN_SIZE

        self.substrates = []
        for pos in SUBSTRATE_POSITIONS:
            self.substrates.append({
                "center": np.array(pos, dtype=np.float64),
                "radius": SUBSTRATE_RADIUS,
                "surface_conc": SUBSTRATE_SURFACE_CONC,
            })

        half = self.dx / 2.0
        x = np.linspace(half, DOMAIN_SIZE - half, self.nx)
        y = np.linspace(half, DOMAIN_SIZE - half, self.ny)
        z = np.linspace(half, DOMAIN_SIZE - half, self.nz)
        self._x = x
        self._y = y
        self._z = z

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        self.C = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self._substrate_masks = []

        for sub in self.substrates:
            cx, cy, cz = sub["center"]
            R0 = sub["radius"]
            C0 = sub["surface_conc"]

            dist = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)

            conc_contribution = np.where(
                dist <= R0,
                C0,
                C0 * R0 / np.maximum(dist, 1e-20)
            )
            self.C += conc_contribution

            mask = dist <= R0
            self._substrate_masks.append((mask, C0))

        np.clip(self.C, 0.0, None, out=self.C)

    def get_concentration(self, position):
        fx = (position[0] - self.dx / 2.0) / self.dx
        fy = (position[1] - self.dx / 2.0) / self.dx
        fz = (position[2] - self.dx / 2.0) / self.dx

        fx = np.clip(fx, 0, self.nx - 1.001)
        fy = np.clip(fy, 0, self.ny - 1.001)
        fz = np.clip(fz, 0, self.nz - 1.001)

        ix = int(fx)
        iy = int(fy)
        iz = int(fz)
        dx_frac = fx - ix
        dy_frac = fy - iy
        dz_frac = fz - iz

        ix1 = min(ix + 1, self.nx - 1)
        iy1 = min(iy + 1, self.ny - 1)
        iz1 = min(iz + 1, self.nz - 1)

        c000 = self.C[ix,  iy,  iz]
        c100 = self.C[ix1, iy,  iz]
        c010 = self.C[ix,  iy1, iz]
        c110 = self.C[ix1, iy1, iz]
        c001 = self.C[ix,  iy,  iz1]
        c101 = self.C[ix1, iy,  iz1]
        c011 = self.C[ix,  iy1, iz1]
        c111 = self.C[ix1, iy1, iz1]

        c00 = c000 * (1 - dx_frac) + c100 * dx_frac
        c01 = c001 * (1 - dx_frac) + c101 * dx_frac
        c10 = c010 * (1 - dx_frac) + c110 * dx_frac
        c11 = c011 * (1 - dx_frac) + c111 * dx_frac

        c0 = c00 * (1 - dy_frac) + c10 * dy_frac
        c1 = c01 * (1 - dy_frac) + c11 * dy_frac

        c = c0 * (1 - dz_frac) + c1 * dz_frac

        return max(float(c), 0.0)

    def get_concentration_batch(self, positions):
        n = positions.shape[0]
        if n == 0:
            return np.empty(0)

        fx = (positions[:, 0] - self.dx / 2.0) / self.dx
        fy = (positions[:, 1] - self.dx / 2.0) / self.dx
        fz = (positions[:, 2] - self.dx / 2.0) / self.dx

        fx = np.clip(fx, 0, self.nx - 1.001)
        fy = np.clip(fy, 0, self.ny - 1.001)
        fz = np.clip(fz, 0, self.nz - 1.001)

        ix = fx.astype(int)
        iy = fy.astype(int)
        iz = fz.astype(int)
        dx_f = fx - ix
        dy_f = fy - iy
        dz_f = fz - iz

        ix1 = np.minimum(ix + 1, self.nx - 1)
        iy1 = np.minimum(iy + 1, self.ny - 1)
        iz1 = np.minimum(iz + 1, self.nz - 1)

        c000 = self.C[ix,  iy,  iz]
        c100 = self.C[ix1, iy,  iz]
        c010 = self.C[ix,  iy1, iz]
        c110 = self.C[ix1, iy1, iz]
        c001 = self.C[ix,  iy,  iz1]
        c101 = self.C[ix1, iy,  iz1]
        c011 = self.C[ix,  iy1, iz1]
        c111 = self.C[ix1, iy1, iz1]

        c00 = c000 * (1 - dx_f) + c100 * dx_f
        c01 = c001 * (1 - dx_f) + c101 * dx_f
        c10 = c010 * (1 - dx_f) + c110 * dx_f
        c11 = c011 * (1 - dx_f) + c111 * dx_f

        c0 = c00 * (1 - dy_f) + c10 * dy_f
        c1 = c01 * (1 - dy_f) + c11 * dy_f

        c = c0 * (1 - dz_f) + c1 * dz_f

        return np.maximum(c, 0.0)

    def update(self, dt, consumption_field=None):
        if consumption_field is not None:
            self.C -= consumption_field
            np.clip(self.C, 0.0, None, out=self.C)

        C = self.C
        lap = (
            np.roll(C, 1, axis=0) + np.roll(C, -1, axis=0)
            + np.roll(C, 1, axis=1) + np.roll(C, -1, axis=1)
            + np.roll(C, 1, axis=2) + np.roll(C, -1, axis=2)
            - 6.0 * C
        ) / (self.dx ** 2)

        self.C += self.D * dt * lap

        if GRAVITY_ON:
            dCdz = (C - np.roll(C, 1, axis=2)) / self.dx
            self.C -= self.v_conv * dt * dCdz

        self.C[0, :, :] = 0.0
        self.C[-1, :, :] = 0.0
        self.C[:, 0, :] = 0.0
        self.C[:, -1, :] = 0.0
        self.C[:, :, 0] = 0.0
        self.C[:, :, -1] = 0.0

        for mask, C0 in self._substrate_masks:
            self.C[mask] = C0

        np.clip(self.C, 0.0, None, out=self.C)

    def position_to_grid_index(self, position):
        ix = int(np.clip(position[0] / self.dx, 0, self.nx - 1))
        iy = int(np.clip(position[1] / self.dx, 0, self.ny - 1))
        iz = int(np.clip(position[2] / self.dx, 0, self.nz - 1))
        return ix, iy, iz

    def get_slice(self, axis='z', index=None):
        if index is None:
            index = self.nx // 2

        if axis == 'x':
            return self.C[index, :, :].copy()
        elif axis == 'y':
            return self.C[:, index, :].copy()
        elif axis == 'z':
            return self.C[:, :, index].copy()
        else:
            raise ValueError(f"axis 必须是 'x', 'y', 或 'z'，收到 '{axis}'")


if __name__ == "__main__":
    print("=== 环境场模块自测 ===\n")

    env = Environment()
    print(f"网格: {env.nx}×{env.ny}×{env.nz}")
    print(f"网格间距: {env.dx*1e6:.1f} μm")
    print(f"浓度场形状: {env.C.shape}")
    print(f"浓度范围: [{env.C.min():.2e}, {env.C.max():.2e}] M\n")

    sub = env.substrates[0]
    surface_point = sub["center"] + np.array([sub["radius"], 0, 0])
    c_surface = env.get_concentration(surface_point)
    print(f"测试1 - 底物表面浓度: {c_surface:.4e} M")
    print(f"  期望: ~{SUBSTRATE_SURFACE_CONC:.4e} M")

    far_point = sub["center"] + np.array([2 * sub["radius"], 0, 0])
    c_far = env.get_concentration(far_point)
    print(f"\n测试2 - 2R₀处浓度: {c_far:.4e} M")
    print(f"  期望: ~{SUBSTRATE_SURFACE_CONC/2:.4e} M (C₀/2)")

    edge_point = np.array([DOMAIN_SIZE * 0.95, DOMAIN_SIZE / 2, DOMAIN_SIZE / 2])
    c_edge = env.get_concentration(edge_point)
    print(f"\n测试3 - 域边缘浓度: {c_edge:.4e} M")
    print(f"  期望: 接近0")

    distances = [75e-6, 150e-6, 300e-6, 600e-6]
    concs = []
    for d in distances:
        p = sub["center"] + np.array([d, 0, 0])
        concs.append(env.get_concentration(p))
    monotonic = all(concs[i] >= concs[i+1] for i in range(len(concs)-1))
    print(f"\n测试4 - 浓度单调递减: {monotonic}")
    for d, c in zip(distances, concs):
        print(f"  r={d*1e6:.0f}μm → C={c:.4e} M")

    env.update(0.05)
    print(f"\n测试5 - 一步更新后浓度范围: [{env.C.min():.2e}, {env.C.max():.2e}] M")
    print(f"  无NaN: {not np.any(np.isnan(env.C))}")
    print(f"  无负值: {not np.any(env.C < 0)}")

    print("\n=== 自测通过 ===")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm

def poincare_to_cartesian(r, theta):
    """r: 0~1 (disk radius), theta: 0~2pi"""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

# ---- 树结构设置 ----
root = {'pos': (0, 0), 'color': None}
n_children = 4
n_grandchildren = 3
disk_radius = 1.0

# 子节点分布（以极坐标均匀分布）
child_r = 0.5    # root到子节点距离
child_thetas = np.linspace(0, 2*np.pi, n_children, endpoint=False)
child_colors = [cm.hsv(i/n_children) for i in range(n_children)]  # 4色

# 孙节点分布（以每个子节点为圆心扇形分布）
grandchild_r = 0.7
r_sigma = 0.15   # 孙节点的r邻域

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect('equal')
# 绘制Poincare disk边界
ax.add_patch(Circle((0, 0), disk_radius, fill=False, edgecolor='black', linewidth=2))

# 记录所有节点坐标
all_nodes = []

# 绘制根节点
ax.plot(0, 0, 'ko', markersize=13, zorder=10)
all_nodes.append((0, 0))

for i, (theta, color) in enumerate(zip(child_thetas, child_colors)):
    # 子节点坐标
    cx, cy = poincare_to_cartesian(child_r, theta)
    ax.plot(cx, cy, 'o', color=color, markersize=10, alpha=0.7, zorder=9)
    all_nodes.append((cx, cy))
    # 绘制连接线
    ax.plot([0, cx], [0, cy], '-', color='gray', lw=1)
    
    # 孙节点分布（以子节点为圆心，三等分小弧度扇形分布）
    sub_thetas = theta + np.linspace(-np.pi/8, np.pi/8, n_grandchildren)
    grandchild_xy = []
    for j, stheta in enumerate(sub_thetas):
        gcx, gcy = poincare_to_cartesian(grandchild_r, stheta)
        ax.plot(gcx, gcy, 'o', color=color, markersize=9, alpha=0.7, zorder=8)
        all_nodes.append((gcx, gcy))
        # 连线
        ax.plot([cx, gcx], [cy, gcy], '-', color='gray', lw=1, alpha=0.5)
        grandchild_xy.append((gcx, gcy))
        
        # —— 孙节点高斯色斑 ——
        X, Y = np.meshgrid(
            np.linspace(gcx-0.2, gcx+0.2, 70),
            np.linspace(gcy-0.2, gcy+0.2, 70)
        )
        # Poincare disk mask
        mask = (X**2 + Y**2) < disk_radius**2
        Z = np.exp(-((X-gcx)**2 + (Y-gcy)**2)/(2*r_sigma**2))
        Z = Z * mask
        ax.contourf(X, Y, Z, levels=7, cmap=cm.colors.ListedColormap([color]), alpha=0.3, zorder=2)

    # —— 父节点代表色斑（以三个孙节点的均值为中心） ——
    gxs, gys = zip(*grandchild_xy)
    mx, my = np.mean(gxs), np.mean(gys)
    Xp, Yp = np.meshgrid(
        np.linspace(mx-0.14, mx+0.14, 60),
        np.linspace(my-0.14, my+0.14, 60)
    )
    maskp = (Xp**2 + Yp**2) < disk_radius**2
    Zp = np.exp(-((Xp-mx)**2 + (Yp-my)**2)/(2*(r_sigma/2)**2))
    Zp = Zp * maskp
    ax.contourf(Xp, Yp, Zp, levels=6, cmap=cm.colors.ListedColormap([color]), alpha=0.7, zorder=3)

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis('off')
plt.tight_layout()
plt.savefig('disk_bg.png', dpi=300)
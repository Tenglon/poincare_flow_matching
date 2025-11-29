import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

# ---------- Poincaré utilities ---------- #
def rho_to_r(rho): return np.tanh(rho/2)

def poincare_circle_euclid(p, rho_h):
    R = rho_to_r(rho_h)
    norm2 = np.dot(p, p)
    denom = 1 - norm2*R*R
    if denom <= 0: return None, None
    return ((1-R*R)/denom)*p, ((1-norm2)*R)/denom

def mobius_add(x, y):
    xy, x2, y2 = np.dot(x, y), np.dot(x, x), np.dot(y, y)
    denom = 1+2*xy+x2*y2
    return ((1+2*xy+y2)*x + (1-x2)*y)/denom

def mobius_scalar_mult(a, x):
    norm = np.linalg.norm(x)
    if norm == 0: return x.copy()
    t = np.tanh(a*np.arctanh(norm))
    return (t/norm)*x

def geodesic(u, v, n=200):
    diff = mobius_add(-u, v)
    ts = np.linspace(0, 1, n)
    return np.array([mobius_add(u, mobius_scalar_mult(t, diff)) for t in ts])

# ---------- Color helpers ---------- #
def lighten(rgb, amt): return tuple(np.clip(np.array(rgb)+(1-np.array(rgb))*amt,0,1))
def darken(rgb, amt): return tuple(np.clip(np.array(rgb)*(1-amt),0,1))
def shift(rgb, delta): return lighten(rgb, delta) if delta>=0 else darken(rgb,-delta)

# ---------- Draw Gaussian via true Poincaré circles ---------- #
def draw_poincare_gaussian(ax, p, sigma, color, alpha_peak, layers=50):
    max_rho = 3*sigma
    for rho in np.linspace(max_rho, 0, layers, endpoint=False):
        alpha = alpha_peak*np.exp(-rho**2/(2*sigma**2))
        if alpha < 0.004: continue
        c, r = poincare_circle_euclid(p, rho)
        if c is None or r <= 1e-4: continue
        if np.linalg.norm(c)-r >= 1: continue
        circ = Circle(c, r, color=color, alpha=alpha, lw=0)
        circ.set_clip_path(disk)
        ax.add_patch(circ)

# ---------- Parameters ---------- #
base_hex = ['#0077BB','#EE7733','#009988','#CC3311']
base_rgb = [mcolors.to_rgb(h) for h in base_hex]

n_children, n_grand = 4, 3
rotation = np.pi/4
child_angles = np.linspace(0, 2*np.pi, n_children, endpoint=False)+rotation

rho_child, rho_grand = 1.2, 2.0
r_child, r_grand = rho_to_r(rho_child), rho_to_r(rho_grand)

sigma_child, sigma_grand = 0.6, 0.4

# --- Increased transparency ---
alpha_child_peak = 0.07   # child Gaussian
alpha_grand_peak = 0.22   # grand Gaussian
child_node_alpha = 0.12
grand_node_alpha = 0.65

variation = [-0.25,0,0.25]

child_r, grand_r = 0.033, 0.02

# ---------- Plot ---------- #
fig, ax = plt.subplots(figsize=(6,6))
disk = Circle((0,0),1,ec='black',fill=False,lw=1.2)
ax.add_patch(disk)
ax.set_clip_path(disk)

for i, th in enumerate(child_angles):
    base = base_rgb[i]
    child_pos = np.array([r_child*np.cos(th), r_child*np.sin(th)])
    # child gaussian
    draw_poincare_gaussian(ax, child_pos, sigma_child, base, alpha_child_peak)
    # child node
    ax.add_patch(Circle(child_pos, child_r, color=base, alpha=child_node_alpha, ec='black', lw=0.8))
    # geodesic root-child
    ax.plot(*geodesic(np.zeros(2), child_pos).T, color='gray', lw=0.8)
    # grandchildren
    for j, dth in enumerate(np.linspace(-np.pi/8, np.pi/8, n_grand)):
        grand_pos = np.array([r_grand*np.cos(th+dth), r_grand*np.sin(th+dth)])
        g_color = shift(base, variation[j])
        draw_poincare_gaussian(ax, grand_pos, sigma_grand, g_color, alpha_grand_peak)
        ax.add_patch(Circle(grand_pos, grand_r, color=g_color, alpha=grand_node_alpha, ec='black', lw=0.5))
        ax.plot(*geodesic(child_pos, grand_pos).T, color='gray', lw=0.5)

# root node
ax.add_patch(Circle((0,0),0.045,color='black',ec='white',lw=0.7))

ax.set_aspect('equal')
ax.set_xlim([-1.05,1.05]); ax.set_ylim([-1.05,1.05])
ax.axis('off')
plt.tight_layout()

path = 'poincare_tree_tmp.png'
plt.savefig(path,dpi=300)
plt.show()
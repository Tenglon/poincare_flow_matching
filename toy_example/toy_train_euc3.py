import os
import shutil
import time
from matplotlib.patches import Circle
import torch
from torchdyn.core import NeuralODE
from torchcfm.model.models import MLP
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher
from utils import torch_wrapper, plot_trajectories_cond2, c_normal_sample2

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np

traj_dir = "trajectory_euc3"
shutil.rmtree(traj_dir, ignore_errors=True)
os.makedirs(traj_dir, exist_ok=True)

# define the model
sigma = 1e-3
dim = 2
batch_size = 1024
model = MLP(dim=dim, time_varying=True).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# FM = ConditionalFlowMatcher(sigma=sigma)
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)  # Use same as toy_train_euc2.py

# Color scheme from fans2.py
base_hex = ['#0077BB','#EE7733','#009988','#CC3311']
base_rgb = [mcolors.to_rgb(h) for h in base_hex]

# Color variation for grandchildren (from fans2.py)
variation = [-0.25, 0, 0.25]

# Color shift functions from fans2.py
def lighten(rgb, amt): 
    return tuple(np.clip(np.array(rgb) + (1 - np.array(rgb)) * amt, 0, 1))
def darken(rgb, amt): 
    return tuple(np.clip(np.array(rgb) * (1 - amt), 0, 1))
def shift(rgb, delta): 
    return lighten(rgb, delta) if delta >= 0 else darken(rgb, -delta)

# Generate 12 colors: 4 parent colors, each with 3 variations
colors_list = []
for base in base_rgb:
    for var in variation:
        shifted = shift(base, var)
        colors_list.append(shifted)
colors = np.array(colors_list)

# Coordinate system from fans2.py
def rho_to_r(rho): 
    return np.tanh(rho / 2)

def poincare_circle_euclid(p, rho_h):
    R = rho_to_r(rho_h)
    norm2 = np.dot(p, p)
    denom = 1 - norm2 * R * R
    if denom <= 0: 
        return None, None
    return ((1 - R * R) / denom) * p, ((1 - norm2) * R) / denom

def draw_poincare_gaussian(ax, p, sigma, color, alpha_peak, layers=50, disk=None):
    """Draw Gaussian via true Poincaré circles"""
    max_rho = 3 * sigma
    for rho in np.linspace(max_rho, 0, layers, endpoint=False):
        alpha = alpha_peak * np.exp(-rho**2 / (2 * sigma**2))
        if alpha < 0.004: 
            continue
        c, r = poincare_circle_euclid(p, rho)
        if c is None or r <= 1e-4: 
            continue
        if np.linalg.norm(c) - r >= 1: 
            continue
        circ = Circle(c, r, color=color, alpha=alpha, lw=0)
        if disk is not None:
            circ.set_clip_path(disk)
        ax.add_patch(circ)

# Parameters from fans2.py
n_children, n_grand = 4, 3
rotation = np.pi / 4
child_angles = np.linspace(0, 2 * np.pi, n_children, endpoint=False) + rotation

rho_child, rho_grand = 1.2, 2.0
r_child = rho_to_r(rho_child)
r_grand = rho_to_r(rho_grand)

sigma_child = 0.6
sigma_grand = 0.4
alpha_child_peak = 0.07
alpha_grand_peak = 0.22

# Compute centers: 12 grandchild nodes (same structure as fans2.py)
centers_list = []
for i, th in enumerate(child_angles):
    for dth in np.linspace(-np.pi/8, np.pi/8, n_grand):
        grand_pos = np.array([r_grand * np.cos(th + dth), r_grand * np.sin(th + dth)])
        centers_list.append(grand_pos)

centers = np.array(centers_list).astype(np.float32)  # 12 grandchild centers

# Training parameters from toy_train_euc2.py
shuffled_index = [10, 3, 1, 7, 11, 5, 9, 0, 8, 4, 6, 2]
selected_pairs_raw = [(0, 2), (3, 4), (7, 8), (10, 11)]
selected_pairs = [(shuffled_index.index(item[0]), shuffled_index.index(item[1])) for item in selected_pairs_raw]
colors_shuffled = colors[shuffled_index]

source_inx = torch.tensor([item[0] for item in selected_pairs])
target_inx = torch.tensor([item[1] for item in selected_pairs])

source_order = [7, 1, 3, 0]  # from toy_train_euc2.py
target_order = [11, 9, 8, 4]  # from toy_train_euc2.py

# Create canvas function using fans2.py style
def get_canvas():
    """Create canvas with Poincaré Gaussians for parent and child nodes"""
    fig, ax = plt.subplots(figsize=(8, 6))  # Wider figure to accommodate legend on the right
    
    # Create disk for clipping
    disk = Circle((0, 0), 1, ec='black', fill=False, lw=1.2)
    ax.add_patch(disk)
    ax.set_clip_path(disk)
    
    # Draw parent Gaussians at child positions (4 parent nodes)
    for i in range(4):
        base_color = base_rgb[i]
        child_pos = np.array([r_child * np.cos(child_angles[i]), r_child * np.sin(child_angles[i])])
        draw_poincare_gaussian(ax, child_pos, sigma_child, base_color, 
                             alpha_child_peak, disk=disk)
    
    # Draw child Gaussians at grandchild positions (12 child nodes)
    # Use shuffled colors to match the training data
    for i in range(12):
        grand_pos = centers[i]
        grand_color = colors_shuffled[i]
        draw_poincare_gaussian(ax, grand_pos, sigma_grand, grand_color, 
                             alpha_grand_peak, disk=disk)
    
    ax.set_aspect('equal')
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.axis('off')
    
    return fig, ax

start = time.time()
for k in range(10000):
    optimizer.zero_grad()

    x0, y0 = c_normal_sample2(batch_size, centers, source_inx, dim=2, var=1e-8)
    x1, y1 = c_normal_sample2(batch_size, centers, target_inx, dim=2, var=1e-8)

    sorted_pairs_y0 = sorted(enumerate(y0), key=lambda x: source_order.index(x[1]))
    sorted_pairs_y1 = sorted(enumerate(y1), key=lambda x: target_order.index(x[1]))

    y0_indices, _ = zip(*sorted_pairs_y0)
    y1_indices, _ = zip(*sorted_pairs_y1)

    y0_indices, y1_indices = torch.tensor(y0_indices), torch.tensor(y1_indices)

    x0 = x0[y0_indices]  # sort the data
    x1 = x1[y1_indices]  # sort the data

    x0, x1 = x0.cuda(), x1.cuda()
    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

    vt = model(torch.cat([xt, t[:, None]], dim=-1))
    loss = torch.mean((vt - ut) ** 2)

    # add a reconstruction loss (from toy_train_euc2.py)
    xt = torch.cat([xt, t[:, None]], dim=-1)

    loss.backward()
    optimizer.step()

    if (k + 1) % 100 == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        start = end
        node = NeuralODE(
            torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )

        source_samples, source_labels = c_normal_sample2(batch_size, centers, source_inx, dim=2, var=1e-8)
        target_samples, target_labels = c_normal_sample2(batch_size, centers, target_inx, dim=2, var=1e-8)
        source_samples = source_samples.cuda()
        with torch.no_grad():
            traj = node.trajectory(
                source_samples,
                t_span=torch.linspace(0, 1, 100).cuda(),
            )
            fig, ax = get_canvas()
            plot_trajectories_cond2(ax, traj.cpu().numpy(), source_labels, target_labels, target_samples, colors_shuffled, centers, 
                                   highlight_prob=0.05)  # 5% of flows will be highlighted with gray color (70% transparency)
            plt.tight_layout()
            fig.savefig(f"{traj_dir}/my_moons_step{k}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

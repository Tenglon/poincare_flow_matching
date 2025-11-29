import os
import shutil
import time
from matplotlib.patches import Circle
import torch
from torchdyn.core import NeuralODE
from torchcfm.model.models import MLP
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher
from utils import torch_wrapper, plot_trajectories_cond2, c_normal_sample2, HypToyData2

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np

traj_dir = "trajectory_hyp3"
shutil.rmtree(traj_dir, ignore_errors=True)
os.makedirs(traj_dir, exist_ok=True)

# define the model
sigma = 1e-3
dim = 2
batch_size = 1024
model = MLP(dim=dim, time_varying=True).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
FM = ConditionalFlowMatcher(sigma=sigma)
# FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

# Initialize HypToyData2 for coordinates and colors
toy_data = HypToyData2()
centers = toy_data.centers.astype(np.float32)  # 12 grandchild centers (ensure float32)
colors = toy_data.colors  # 12 colors
base_rgb = toy_data.base_rgb  # 4 base colors
child_positions = toy_data.child_positions  # 4 child positions

# Parameters from fans2.py
sigma_child = 0.6
sigma_grand = 0.4
alpha_child_peak = 0.07
alpha_grand_peak = 0.22

# Poincaré utilities from fans2.py
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
        child_pos = child_positions[i]
        draw_poincare_gaussian(ax, child_pos, sigma_child, base_color, 
                              alpha_child_peak, disk=disk)
    
    # Draw child Gaussians at grandchild positions (12 child nodes)
    for i in range(4):
        # Get the 3 grandchild positions for this parent
        for j in range(3):
            grand_idx = i * 3 + j
            grand_pos = centers[grand_idx]
            grand_color = colors[grand_idx]
            draw_poincare_gaussian(ax, grand_pos, sigma_grand, grand_color, 
                                 alpha_grand_peak, disk=disk)
    
    ax.set_aspect('equal')
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.axis('off')
    
    return fig, ax

# Select pairs for flow matching (similar to toy_train_hyp2.py)
# Using indices from the 12 grandchild centers
# Each parent has 3 children, so we select pairs within each parent group
selected_pairs_raw = [(0, 2), (3, 4), (7, 8), (10, 11)]  # pairs within each parent group
source_inx = torch.tensor([item[0] for item in selected_pairs_raw])
target_inx = torch.tensor([item[1] for item in selected_pairs_raw])

source_order = [0, 3, 7, 10]
target_order = [2, 4, 8, 11]

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
            plot_trajectories_cond2(ax, traj.cpu().numpy(), source_labels, target_labels, target_samples, colors, centers, 
                                   highlight_prob=0.05)  # 5% of flows will be highlighted with gray color (70% transparency)
            plt.tight_layout()
            fig.savefig(f"{traj_dir}/my_moons_step{k}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)


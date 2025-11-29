import os
import shutil
import time
from matplotlib.patches import Wedge
import torch
from torchdyn.core import NeuralODE
from torchcfm.model.models import MLP
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher
from utils import torch_wrapper, plot_trajectories, plot_trajectories_cond2, c_normal_sample2

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

traj_dir = "trajectory_hyp"
shutil.rmtree(traj_dir, ignore_errors=True)
os.makedirs(traj_dir, exist_ok=True)

# define the model
sigma = 1e-3
dim = 2
batch_size = 1024
model = MLP(dim=dim, time_varying=True).cuda()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
FM = ConditionalFlowMatcher(sigma=sigma)
# FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

# Create colors
color_inx = [4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
cm = mpl.colormaps['tab20b']
colors = cm(np.linspace(0, 1, 20))
colors = colors[color_inx]

# hyperparameters
num_fans = 12
fan_center_loc_radius = 0.45
radius = 1.0
margin = 0.26
angular_margin_factor = 0.7
wedge_length_factor = 0.5
sample_position_factor = 1.8
shuffled_index = list(range(num_fans))
selected_pairs_raw = [(0, 2), (3, 4), (7, 8), (10, 11)]
selected_pairs = [(shuffled_index.index(item[0]), shuffled_index.index(item[1])) for item in selected_pairs_raw]
colors_shuffled = colors[shuffled_index]

# Generate theta values
theta = np.linspace(0, 2 * np.pi, num_fans, endpoint=False, dtype=np.float32)

# Compute x and y values
centers = fan_center_loc_radius * np.vstack([np.cos(theta), np.sin(theta)]).T
# centers = torch.from_numpy(centers)

# Calculate the angle for each fan
theta = np.linspace(0.0 - margin, 2 * np.pi - margin, num_fans, endpoint=False)

# Create a figure and axis
def get_canvas(colors, num_fans, radius, angular_margin_factor, wedge_length_factor, shuffled_index, theta, centers):
    fig, ax = plt.subplots()
    for i in range(num_fans):
    # Define the start and end points for each fan segment
        angular_margin = angular_margin_factor * np.pi / num_fans
        start_angle = theta[i] - angular_margin
        end_angle = theta[i] + 2 * np.pi / num_fans + angular_margin
        wedge = Wedge(centers[i], wedge_length_factor * radius, np.degrees(start_angle), np.degrees(end_angle), facecolor=colors_shuffled[i], alpha=0.5, linewidth=0.1, edgecolor='k')
        ax.add_patch(wedge)
    return fig, ax


source_inx = torch.tensor([item[0] for item in selected_pairs])
target_inx = torch.tensor([item[1] for item in selected_pairs])

# # ------------ testing
# # generate data 
# x0, multinomial_label_y0 = c_normal_sample2(batch_size, centers, source_inx, dim=2, var=1e-8)
# x1, multinomial_label_y1 = c_normal_sample2(batch_size, centers, target_inx, dim=2, var=1e-8)

# # rearrange the data
# x0 = x0[multinomial_label_y0.argsort()]
# x1 = x1[multinomial_label_y1.argsort()]
# fig, ax = get_canvas(colors, num_fans, radius, angular_margin_factor, wedge_length_factor, shuffled_index, theta, centers)

# # test: visualize the data in a temp.png
# plt.scatter(x0[:, 0], x0[:, 1], s=10, alpha=0.8, c='black')
# plt.savefig("temp1.png")
# plt.scatter(x1[:, 0], x1[:, 1], s=10, alpha=0.8, c='blue')
# plt.savefig("temp2.png")
# plt.plot([centers[source_inx[0]][0], centers[target_inx[0]][0]], [centers[source_inx[0]][1], centers[target_inx[0]][1]], color='black', linewidth=0.5, alpha=1)
# plt.plot([centers[source_inx[1]][0], centers[target_inx[1]][0]], [centers[source_inx[1]][1], centers[target_inx[1]][1]], color='black', linewidth=0.5, alpha=1)
# plt.plot([centers[source_inx[2]][0], centers[target_inx[2]][0]], [centers[source_inx[2]][1], centers[target_inx[2]][1]], color='black', linewidth=0.5, alpha=1)
# plt.plot([centers[source_inx[3]][0], centers[target_inx[3]][0]], [centers[source_inx[3]][1], centers[target_inx[3]][1]], color='black', linewidth=0.5, alpha=1)
# plt.savefig("temp3.png")
# exit()
# # ------------ testing


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

    x0 = x0[y0_indices] # sort the data
    x1 = x1[y1_indices] # sort the data

    x0, x1 = x0.cuda(), x1.cuda()
    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

    vt = model(torch.cat([xt, t[:, None]], dim=-1))
    loss = torch.mean((vt - ut) ** 2)

    # add a reconstruction loss
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
            # plot_trajectories_cond(traj.cpu().numpy(), source_labels, target_labels, target_samples, k, traj_dir)
            fig, ax = get_canvas(colors, num_fans, radius, angular_margin_factor, wedge_length_factor, shuffled_index, theta, centers)
            plot_trajectories_cond2(ax, traj.cpu().numpy(), source_labels, target_labels, target_samples, colors_shuffled, centers)
            fig.savefig(f"{traj_dir}/my_moons_step{k}.png")
            plt.close(fig)
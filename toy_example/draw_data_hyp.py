
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib as mpl
import numpy as np
import torch

def c_normal_sample(n, centers, dim = 2, var=1):
    """Sample from c normal distributions.
    n: number of samples
    centers: centers of the c distributions
    dim: dimension of each sample
    var: variance of each distribution
    """

    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )

    noise = m.sample((n,))
    label = torch.multinomial(torch.ones(centers.shape[0]), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[label[i]] + noise[i])
    data = torch.stack(data)

    return data, label

# Number of fans you want
num_fans = 12
color_inx = [4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]

# Create colors
cm = mpl.colormaps['tab20b']
colors = cm(np.linspace(0, 1, 20))
colors = colors[color_inx]

# hyperparameters
fan_center_loc_radius = 0.45
radius = 1.0
margin = 0.26
angular_margin_factor = 0.7
wedge_length_factor = 0.5
sample_position_factor = 1.8

# Create a figure and axis
fig, ax = plt.subplots()

# Define the center and radius of the disk
# Generate theta values
theta = np.linspace(0, 2 * np.pi, num_fans, endpoint=False)

# Compute x and y values
centers = fan_center_loc_radius * np.vstack([np.cos(theta), np.sin(theta)]).T

# Calculate the angle for each fan
theta = np.linspace(0.0 - margin, 2 * np.pi - margin, num_fans, endpoint=False)

index = np.arange(num_fans)
# np.random.shuffle(index)
# print(index)

for i in range(num_fans):
    # Define the start and end points for each fan segment
    angular_margin = angular_margin_factor * np.pi / num_fans
    start_angle = theta[i] - angular_margin
    end_angle = theta[i] + 2 * np.pi / num_fans + angular_margin
    wedge = Wedge(centers[i], wedge_length_factor * radius, np.degrees(start_angle), np.degrees(end_angle), facecolor=colors[index[i]], alpha=0.5, linewidth=0.1, edgecolor='k')
    ax.add_patch(wedge)

for i in range(4):
    # Define the start and end points for each fan segment
    start_angle = theta[i*3] + margin - margin
    end_angle = theta[i*3+2] + margin + margin
    wedge = Wedge((0,0), radius, np.degrees(start_angle), np.degrees(end_angle), facecolor=colors[index[i*3]], alpha=0.3, linewidth=0.1, edgecolor='k')
    ax.add_patch(wedge)

samples, labels = c_normal_sample(2048 , torch.tensor(centers) * sample_position_factor, var=1e-8)

plt.scatter(samples[:, 0], samples[:, 1], s=1, c=colors[labels], alpha=1, marker='o', linewidths=0.1, edgecolors='k')
# Set equal aspect and show the plot
ax.set_aspect('equal')
ax.set_xlim(-radius, radius)
ax.set_ylim(-radius, radius)
ax.axis("off")
# plt.show()
plt.savefig("data_vis_hyp.pdf", dpi=600)
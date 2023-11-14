
from itertools import combinations
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib as mpl
import numpy as np
import torch
from utils import c_normal_sample, generate_targets_mean, softmax
from geodisc import geodesic_fn

def softmax(x, T = 1):
    x = x / T
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# Number of fans you want
num_fans = 12
color_inx = [4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]

# Create colors
cm = mpl.colormaps['tab20b']
colors = cm(np.linspace(0, 1, 20))
colors = colors[color_inx]

hierarchy = [[4, 5, 6], [8, 9, 10], [12, 13, 14], [16, 17, 18]]

# shuffled_inx = [10, 3, 1, 7, 11, 5, 9, 0, 8, 4, 6, 2]
# shuffled_inx = [1, 3, 10, 4, 11, 2, 8, 5, 7, 0, 6, 9]
# shuffled_inx = [7, 0, 6, 3, 9, 2, 5, 1, 4, 11, 8, 10]
# selected_pairs = [(0, 2), (3, 4), (7, 8), (10, 11)]
# selected_pairs = [(0, 1), (4, 5), (6, 8), (9, 11)]
# selected_pairs = [(0, 1), (3, 4), (6, 8), (9, 10)]
selected_pairs = [(0, 2), (3, 5), (7, 8), (9, 11)]


# hyperparameters
fan_center_loc_radius = 0.45
radius = 1.0
margin = 0.26
angular_margin_factor = 0.7
wedge_length_factor = 0.5
sample_position_factor = 1.8

# Create a figure and axis
fig, ax = plt.subplots()

# Generate theta values
theta = np.linspace(0, 2 * np.pi, num_fans, endpoint=False, dtype=np.float32)

# Compute x and y values
centers = fan_center_loc_radius * np.vstack([np.cos(theta), np.sin(theta)]).T

# Calculate the angle for each fan
theta = np.linspace(0.0 - margin, 2 * np.pi - margin, num_fans, endpoint=False)

for i in range(num_fans):
    # Define the start and end points for each fan segment
    angular_margin = angular_margin_factor * np.pi / num_fans
    start_angle = theta[i] - angular_margin
    end_angle = theta[i] + 2 * np.pi / num_fans + angular_margin
    wedge = Wedge(centers[i], wedge_length_factor * radius, np.degrees(start_angle), np.degrees(end_angle), facecolor=colors[i], alpha=0.5, linewidth=0.1, edgecolor='k')
    ax.add_patch(wedge)

# Set equal aspect and show the plot
ax.set_aspect('equal')
ax.set_xlim(-radius, radius)
ax.set_ylim(-radius, radius)
ax.axis("off")

fan_centers = 1.3 * centers

for pair in selected_pairs:

    # connect two fans with a line
    x0, y0 = fan_centers[pair[0], 0], fan_centers[pair[0], 1]
    x1, y1 = fan_centers[pair[1], 0], fan_centers[pair[1], 1]
    # plt.plot([x0, x1], [y0, y1], c='k', linewidth=2.5)    

    # connect two fans with 1000 points
    # x = np.linspace(x0, x1, 1000)
    # y = np.linspace(y0, y1, 1000)
    points = geodesic_fn(np.array([x0, y0]), np.array([x1, y1]), 1000)
    x, y = points[:, 0], points[:, 1]

    # # calculate the distance to each of the 12 fan centers
    dist = np.zeros((1000, 12))
    for i in range(12):
        dist[:, i] = np.sqrt((x - fan_centers[i, 0])**2 + (y - fan_centers[i, 1])**2)
    # normalize the distance to the range [0, 1] using softmax
    dist = softmax(-dist, T=0.02)

    # dist /= np.sum(dist, axis=1, keepdims=True)
    color_line = dist@colors
    color_line[color_line > 1] = 1

    plt.scatter(x, y, c=color_line, s = 1, alpha=0.2)
    
    # put two markers for the two centers
    plt.scatter(x0, y0, s=10, c='k')
    plt.scatter(x1, y1, s=10, c='k')

plt.savefig(f"test_hyp.png", dpi=300)
plt.close()
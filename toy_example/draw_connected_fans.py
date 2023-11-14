
from itertools import combinations
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib as mpl
import numpy as np
import torch
from utils import c_normal_sample, generate_targets_mean

# Number of fans you want
num_fans = 12
color_inx = [4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]

# Create colors
cm = mpl.colormaps['tab20b']
colors = cm(np.linspace(0, 1, 20))
colors = colors[color_inx]

# Construct a valid index
seed = 45
for seed in [16, 94, 95]:
    np.random.seed(seed)
    index = np.arange(num_fans)
    np.random.shuffle(index)

    valid_flag = False
    # Flag to check if the current index is valid
    while not valid_flag:

        valid_flag = True
        np.random.shuffle(index)

        for i in range(num_fans):
            if abs(index[i] - index[(i + 1) % num_fans]) == 1 or abs(index[i] - index[(i + 1) % num_fans]) == num_fans - 1:
                valid_flag = False
                break

    print(index)

    index = index.tolist()
    hierarchy = [[index.index(i) for i in range(3)], 
                [index.index(i) for i in range(3, 6)], 
                [index.index(i) for i in range(6, 9)], 
                [index.index(i) for i in range(9, 12)]]

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
        wedge = Wedge(centers[i], wedge_length_factor * radius, np.degrees(start_angle), np.degrees(end_angle), facecolor=colors[index[i]], alpha=0.5, linewidth=0.1, edgecolor='k')
        ax.add_patch(wedge)

    # Set equal aspect and show the plot
    ax.set_aspect('equal')
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.axis("off")
    # plt.show()

    fan_centers = 1.2 * fan_center_loc_radius * np.vstack([np.cos(theta), np.sin(theta)]).T
    # connect the two fans within each cluster
    for sub_tree in hierarchy:
        pairs = list(combinations(sub_tree, 2))
        # random pick one pair
        inx = np.random.randint(0, len(pairs))
        pair = pairs[inx]
        # connect two centers with a line
        x0, y0 = fan_centers[pair[0], 0], fan_centers[pair[0], 1]
        x1, y1 = fan_centers[pair[1], 0], fan_centers[pair[1], 1]
        plt.plot([x0, x1], [y0, y1], c='k', linewidth=0.5)    

    plt.savefig(f"test_{seed}.png", dpi=300)
    plt.close()

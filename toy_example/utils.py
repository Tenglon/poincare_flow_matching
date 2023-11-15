import matplotlib as mpl
from matplotlib.patches import Wedge
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from itertools import combinations

def softmax(x, T = 1):
    x = x / T
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / np.sum(e_x, axis=1, keepdims=True)

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))
    

def plot_trajectories(traj, k, save_dir):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    # plt.show()
    plt.savefig(f"{save_dir}/my_moons_step{k}.png")

def plot_trajectories_cond(traj, source_labels, target_labels, target_samples, k, save_dir):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    # show text on the plot for the labels
    unique_labels = torch.unique(source_labels)
    for label in unique_labels:
        pos = torch.where(source_labels == label)[0][0]
        plt.text(traj[0, pos, 0] + 0.01, traj[0, pos, 1] + 0.01, str(label.item()), fontsize=10, color="black")

    # show text on the plot for the labels
    unique_labels = torch.unique(target_labels)
    for label in unique_labels:
        pos = torch.where(target_labels == label)[0][0]
        plt.text(target_samples[pos, 0] + 0.01, target_samples[pos, 1] + 0.01, str(label.item()), fontsize=10, color="blue")

    plt.scatter(target_samples[:, 0], target_samples[:, 1], s= 1, alpha=0.1, c="grey")

    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=1, alpha=0.5, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=1, alpha=0.5, c="blue")
    plt.legend(["Prior z(S)", "Flow", "z(0)"])
    # set range to be in -1, 1
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    plt.xticks([])
    plt.yticks([])
    # plt.show()
    plt.savefig(f"{save_dir}/my_moons_step{k}.png")

def plot_trajectories_cond2(ax, traj, source_labels, target_labels, target_samples):
    """Plot trajectories of some selected samples."""
    n = 2000
    # show text on the plot for the labels
    unique_labels = torch.unique(source_labels)
    for label in unique_labels:
        pos = torch.where(source_labels == label)[0][0]
        ax.text(traj[0, pos, 0] + 0.03, traj[0, pos, 1] + 0.03, str(label.item()), fontsize=10, color="black")

    # show text on the plot for the labels
    unique_labels = torch.unique(target_labels)
    for label in unique_labels:
        pos = torch.where(target_labels == label)[0][0]
        ax.text(target_samples[pos, 0] + 0.03, target_samples[pos, 1] + 0.03, str(label.item()), fontsize=10, color="blue")

    ax.scatter(target_samples[:, 0], target_samples[:, 1], s= 1, alpha=0.1, c="grey")

    ax.scatter(traj[0, :n, 0], traj[0, :n, 1], s=1, alpha=0.5, c="black")
    ax.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    ax.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=1, alpha=0.5, c="blue")
    ax.legend(["Prior z(S)", "Flow", "z(0)"])
    # set range to be in -1, 1
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.set_xticks([])
    ax.set_yticks([])
    # plt.show()

def c_normal_sample(n, centers, dim = 2, var=1):
    """Sample from c normal distributions.
    n: number of samples
    centers: centers of the c distributions
    dim: dimension of each sample
    var: variance of each distribution
    """
    centers = torch.tensor(centers)
    if len(centers.shape) == 1:
        centers = centers.unsqueeze(0)

    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )

    noise = m.sample((n,))
    multinomial_label = torch.multinomial(torch.ones(centers.shape[0]), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multinomial_label[i]] + noise[i])
    data = torch.stack(data)

    return data, multinomial_label

def c_normal_sample2(n, centers, class_inx, dim = 2, var=1):
    """Sample from c normal distributions.
    n: number of samples
    centers: centers of the c distributions
    dim: dimension of each sample
    var: variance of each distribution
    """
    centers = torch.tensor(centers)
    centers = centers[class_inx]
    if len(centers.shape) == 1:
        centers = centers.unsqueeze(0)

    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )

    noise = m.sample((n,))
    multinomial_label = torch.multinomial(torch.ones(centers.shape[0]), n, replacement=True)
    label = class_inx[multinomial_label]
    data = []
    for i in range(n):
        data.append(centers[multinomial_label[i]] + noise[i])
    data = torch.stack(data)

    return data, label

def generate_targets_mean(centers, hierarchy, n_samples=1024, var=1e-8):

    samples = []
    labels = []
    n_clusters = len(hierarchy)
    for sub_tree in hierarchy:
        sub_centers = [centers[category] for category in sub_tree]
        sub_centers = torch.stack(sub_centers)
        center = sub_centers.mean(dim=0, keepdim=True)

        cluster_samples, _ = c_normal_sample(n_samples // n_clusters, center, dim=2, var=var)
        samples.append(cluster_samples)
        labels.append(torch.tensor(sub_tree).repeat(n_samples // n_clusters, 1))
    
    samples = torch.cat(samples, dim = 0)
    labels = torch.cat(labels, dim = 0)

    return samples, labels


def get_oppo_pair(closest_diff, sub_tree):
    pairs = combinations(sub_tree, 2)

        # Iterate over all pairs to find the one with a difference closest to 6
    for pair in pairs:
        current_diff = abs(pair[0] - pair[1])
        if abs(current_diff - 5) <= abs(closest_diff - 5):
            closest_diff = current_diff
            closest_pair = pair

    return closest_pair

class Toydata:

    def __init__(self):

        self.num_fans = 12
        self.color_inx = [4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]

        # Hyperparameters
        self.fan_center_loc_radius = 0.45
        self.radius = 1.0
        self.margin = 0.26
        self.angular_margin_factor = 0.7
        self.wedge_length_factor = 0.5
        self.sample_position_factor = 1.8

        # Create colors
        cm = mpl.colormaps['tab20b']
        colors = cm(np.linspace(0, 1, 20))
        self.colors = colors[self.color_inx]

        # Compute root point for each fan
        theta = np.linspace(0, 2 * np.pi, self.num_fans, endpoint=False, dtype=np.float32)
        self.centers = self.fan_center_loc_radius * np.vstack([np.cos(theta), np.sin(theta)]).T

        # Compute the span for each fan
        self.theta = np.linspace(0.0 - self.margin, 2 * np.pi - self.margin, self.num_fans, endpoint=False)

    def draw_fans(self, ax):

        for i in range(self.num_fans):
            # Define the start and end points for each fan segment
            angular_margin = self.angular_margin_factor * np.pi / self.num_fans
            start_angle = self.theta[i] - angular_margin
            end_angle = self.theta[i] + 2 * np.pi / self.num_fans + angular_margin
            wedge = Wedge(self.centers[i], self.wedge_length_factor * self.radius, np.degrees(start_angle), np.degrees(end_angle), facecolor=self.colors[self.index[i]], alpha=0.5, linewidth=0.1, edgecolor='k')
            ax.add_patch(wedge)

    def get_source_samples(self, n_samples=2048):
        source_samples, labels = c_normal_sample(n_samples , torch.tensor(self.centers) * self.sample_position_factor, var=1e-6)
        return source_samples, labels
    
    def get_target_samples(self, n_samples=2048):
        """Generate target samples from the centers of the fans"""
        target_samples, labels = generate_targets_mean(torch.tensor(self.centers), self.hierarchy, n_samples=n_samples, var=1e-6)
        return target_samples, labels
    
    def generate_opposite_samples(self, n_samples=1024, var=1e-8):

        source_samples_list, target_samples_list = [], []
        source_labels_list, target_labels_list = [], []
        n_clusters = len(self.hierarchy) * 2# maybe buggy: hard coded

        # Initialize variables to track the closest difference and corresponding pair
        closest_diff = float('inf')
        closest_pair = None

        for sub_tree in self.hierarchy:
            closest_pair = get_oppo_pair(closest_diff, sub_tree)
            source, target = closest_pair[0], closest_pair[1]

            # print(f"source: {source}, target: {target}")
            # print(f"centers: {self.centers[source]}, {self.centers[target]}")

            source_samples, _ = c_normal_sample(n_samples // n_clusters, self.centers[source], dim=2, var=var)
            target_samples, _ = c_normal_sample(n_samples // n_clusters, self.centers[target], dim=2, var=var)

            # print(f"source_mean: {source_samples.mean(dim=0)}, target_mean: {target_samples.mean(dim=0)}")

            source_samples_list.append(source_samples)
            target_samples_list.append(target_samples)
            source_labels_list.append(torch.tensor(source).repeat(n_samples // n_clusters, 1))
            target_labels_list.append(torch.tensor(target).repeat(n_samples // n_clusters, 1))

        source_samples = torch.cat(source_samples_list, dim = 0)
        target_samples = torch.cat(target_samples_list, dim = 0)
        source_labels = torch.cat(source_labels_list, dim = 0)
        target_labels = torch.cat(target_labels_list, dim = 0)

        return source_samples, target_samples, source_labels, target_labels

    def draw_samples(self, ax, manifold:str):

        source_samples, labels = self.get_source_samples()
        target_samples = self.get_target_samples()

        plt.scatter(source_samples[:, 0], source_samples[:, 1], s=1, c=self.colors[labels], alpha=1, marker='o', linewidths=0.1, edgecolors='k')
        plt.scatter(target_samples[:, 0], target_samples[:, 1], s=2, c='k', alpha=1, marker='o', linewidths=0.1, edgecolors='k')
        # Set equal aspect and show the plot
        ax.set_aspect('equal')
        ax.set_xlim(-self.radius, self.radius)
        ax.set_ylim(-self.radius, self.radius)
        ax.axis("off")
        # plt.show()
        plt.savefig(f"data_vis_{manifold}.pdf", dpi=600)

class HypToyData(Toydata):

    def __init__(self):

        super().__init__()

        self.index = np.arange(self.num_fans)
        # here self.hierarchy means the position of the class on the circle
        self.hierarchy = [[0 ,1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]

    def draw_parent_fans(self, ax):

        for i in range(4):
            # Define the start and end points for each fan segment
            start_angle = self.theta[i*3] + self.margin - self.margin
            end_angle = self.theta[i*3+2] + self.margin + self.margin
            wedge = Wedge((0,0), self.radius, np.degrees(start_angle), np.degrees(end_angle), facecolor=self.colors[self.index[i*3]], alpha=0.3, linewidth=0.1, edgecolor='k')
            ax.add_patch(wedge)

    def draw_samples(self, ax):
        super().draw_samples(ax, "hyp")

class EucToyData(Toydata):

    def __init__(self):

        super().__init__()

        np.random.seed(48) # 43, 45, 48. 49, 50
        self.index = np.arange(self.num_fans)
        np.random.shuffle(self.index)

        valid_flag = False
        # Flag to check if the current index is valid
        while not valid_flag:
            valid_flag = True
            np.random.shuffle(self.index)

            for i in range(self.num_fans):
                if abs(self.index[i] - self.index[(i + 1) % self.num_fans]) == 1 or abs(self.index[i] - self.index[(i + 1) % self.num_fans]) == self.num_fans - 1:
                    valid_flag = False
                    break

        self.index = self.index.tolist()
        # here self.hierarchy means the position of the class on the circle
        self.hierarchy = [[self.index.index(i) for i in range(3)], 
                    [self.index.index(i) for i in range(3, 6)], 
                    [self.index.index(i) for i in range(6, 9)], 
                    [self.index.index(i) for i in range(9, 12)]]
        
    def draw_parent_fans(self, ax):

        for i in range(4):
            # Define the start and end points for each fan segment
            start_angle = self.theta[i*3] + self.margin - self.margin
            end_angle = self.theta[i*3+2] + self.margin + self.margin
            wedge = Wedge((0,0), self.radius, np.degrees(start_angle), np.degrees(end_angle), facecolor='grey', alpha=0.3, linewidth=0.1, edgecolor='k')
            ax.add_patch(wedge)

    def draw_samples(self, ax):
        super().draw_samples(ax, "euc")


if __name__ == "__main__":

    # Test EucToyData
    toy_data = EucToyData()
    fig, ax = plt.subplots()
    toy_data.draw_fans(ax)
    toy_data.draw_parent_fans(ax)
    toy_data.draw_samples(ax)

    # Test HypToyData
    toy_data = HypToyData()
    fig, ax = plt.subplots()
    toy_data.draw_fans(ax)
    toy_data.draw_parent_fans(ax)
    toy_data.draw_samples(ax)
import matplotlib as mpl
from matplotlib.patches import Wedge
import numpy as np
import torch
import math
import matplotlib.pyplot as plt

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

def generate_targets(centers, hierarchy, n_samples=1024, var=1e-8):

    samples = []
    n_clusters = len(hierarchy)
    for sub_tree in hierarchy:
        sub_centers = [centers[category] for category in sub_tree]
        sub_centers = torch.stack(sub_centers)
        center = sub_centers.mean(dim=0, keepdim=True)

        cluster_samples, _ = c_normal_sample(n_samples // n_clusters, center, dim=2, var=var)
        samples.append(cluster_samples)
    
    samples = torch.cat(samples, dim=0)

    return samples

class Toydata:

    def __init__(self):

        self.num_fans = 12
        self.color_inx = [4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]

        # hyperparameters
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

    def draw_samples(self, ax, manifold:str):

        samples, labels = c_normal_sample(2048 , torch.tensor(self.centers) * self.sample_position_factor, var=1e-8)
        target_samples = generate_targets(torch.tensor(self.centers), self.hierarchy, n_samples=2048, var=1e-6)

        plt.scatter(samples[:, 0], samples[:, 1], s=1, c=self.colors[labels], alpha=1, marker='o', linewidths=0.1, edgecolors='k')
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

        np.random.seed(45) # 43, 45, 48. 49, 50
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

    toy_data = EucToyData()
    fig, ax = plt.subplots()
    toy_data.draw_fans(ax)
    toy_data.draw_parent_fans(ax)
    toy_data.draw_samples(ax)

    toy_data = HypToyData()
    fig, ax = plt.subplots()
    toy_data.draw_fans(ax)
    toy_data.draw_parent_fans(ax)
    toy_data.draw_samples(ax)
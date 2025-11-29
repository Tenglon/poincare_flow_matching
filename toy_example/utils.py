import matplotlib as mpl
from matplotlib.patches import Wedge, Circle
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

def plot_trajectories_cond2(ax, traj, source_labels, target_labels, target_samples, colors, centers, highlight_prob=0.05, highlight_colors=None):
    """Plot trajectories of some selected samples.
    
    Args:
        highlight_prob: Probability for each flow to be highlighted (default: 0.05, i.e., 5%)
        highlight_colors: List of colors to use for highlighting (default: gray)
    """
    # Use actual data size instead of hardcoded value
    n = min(2000, traj.shape[1])  # Use actual number of samples, capped at 2000
    # Category labels are not displayed on the plot (removed as requested)

    ax.scatter(target_samples[:, 0], target_samples[:, 1], s= 1, alpha=0.1, c="grey")

    # Store handles and labels for legend
    legend_handles = []
    legend_labels = []
    
    # Add start point (black)
    start_handle = ax.scatter([], [], s=50, alpha=0.8, c="black", marker='o')
    legend_handles.append(start_handle)
    legend_labels.append("Start")
    
    # Add end point (blue)
    end_handle = ax.scatter([], [], s=50, alpha=0.8, c="blue", marker='o')
    legend_handles.append(end_handle)
    legend_labels.append("End")
    
    # Draw actual points
    ax.scatter(traj[0, :n, 0], traj[0, :n, 1], s=1, alpha=0.5, c="black")
    ax.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=1, alpha=0.5, c="blue")

    # Default highlight color: gray (unified color for all highlighted flows)
    if highlight_colors is None:
        highlight_colors = ['gray']  # Unified gray color
    
    # Randomly select which flows to highlight (use actual data size)
    np.random.seed(42)  # For reproducibility
    n_flows = n  # Use actual number of samples to plot
    highlight_mask = np.random.rand(n_flows) < highlight_prob
    
    # connect two fans with n points
    for i in range(traj.shape[0]):

        x = traj[i, :n, 0]
        y = traj[i, :n, 1]

        # calculate the distance to each of the 12 fan centers
        dist = np.zeros((x.shape[0], 12))
        for j in range(12):
            dist[:, j] = np.sqrt((x - centers[j, 0])**2 + (y - centers[j, 1])**2)
        # normalize the distance to the range [0, 1] using softmax
        dist = softmax(-dist, T=0.02)

        # dist /= np.sum(dist, axis=1, keepdims=True)
        color_line = dist@colors
        color_line[color_line > 1] = 1

        # Draw normal flows with low alpha
        normal_mask = ~highlight_mask
        if np.any(normal_mask):
            # Use slice first, then boolean indexing
            traj_slice = traj[i, :n]
            ax.scatter(traj_slice[normal_mask, 0], traj_slice[normal_mask, 1], 
                      s=0.2, alpha=0.2, c=color_line[normal_mask])
        
        # Draw highlighted flows with higher alpha and size (unified gray color, 70% transparency)
        if np.any(highlight_mask):
            # Use slice first, then boolean indexing
            traj_slice = traj[i, :n]
            # Use unified gray color for all highlighted flows
            highlight_color = highlight_colors[0]  # Use first color (gray)
            ax.scatter(traj_slice[highlight_mask, 0], traj_slice[highlight_mask, 1], 
                      s=1.0, alpha=0.7, c=highlight_color)

    # Add class labels to legend (Class1-1, 1-2, 1-3, 2-1, 2-2, 2-3, etc.)
    # 4 parent classes, each with 3 child classes
    for parent_idx in range(4):
        for child_idx in range(3):
            class_idx = parent_idx * 3 + child_idx
            if class_idx < len(colors):
                class_color = colors[class_idx]
                # Ensure color is in correct format (RGB tuple or array)
                if isinstance(class_color, np.ndarray):
                    class_color = tuple(class_color)
                class_handle = ax.scatter([], [], s=50, alpha=0.8, c=[class_color], marker='s')
                legend_handles.append(class_handle)
                legend_labels.append(f"Class{parent_idx+1}-{child_idx+1}")

    # Create legend on the right side outside the plot
    ax.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1.0, 0.5), 
              frameon=True, fancybox=True, shadow=True, fontsize=9)
    
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


class HypToyData2(Toydata):
    """HypToyData2 using color scheme and coordinates from fans2.py"""

    def __init__(self):

        super().__init__()

        # Color scheme from fans2.py
        base_hex = ['#0077BB','#EE7733','#009988','#CC3311']
        base_rgb = [mpl.colors.to_rgb(h) for h in base_hex]
        
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
        self.colors = []
        for base in base_rgb:
            for var in variation:
                shifted = shift(base, var)
                self.colors.append(shifted)
        self.colors = np.array(self.colors)

        # Coordinate system from fans2.py
        def rho_to_r(rho): 
            return np.tanh(rho / 2)
        
        rho_child, rho_grand = 1.2, 2.0
        r_child = rho_to_r(rho_child)
        r_grand = rho_to_r(rho_grand)
        
        n_children, n_grand = 4, 3
        rotation = np.pi / 4
        child_angles = np.linspace(0, 2 * np.pi, n_children, endpoint=False) + rotation
        
        # Compute centers: 4 child nodes + 12 grandchild nodes
        centers_list = []
        
        # Child nodes (4 centers)
        for th in child_angles:
            child_pos = np.array([r_child * np.cos(th), r_child * np.sin(th)])
            centers_list.append(child_pos)
        
        # Grandchild nodes (12 centers)
        for i, th in enumerate(child_angles):
            base = base_rgb[i]
            for dth in np.linspace(-np.pi/8, np.pi/8, n_grand):
                grand_pos = np.array([r_grand * np.cos(th + dth), r_grand * np.sin(th + dth)])
                centers_list.append(grand_pos)
        
        self.centers = np.array(centers_list)
        
        # Index and hierarchy: same structure as HypToyData
        self.index = np.arange(self.num_fans)
        # Hierarchy: 4 groups of 3 (child nodes are indices 0-3, grandchildren are 4-15, but we only use 12)
        # Actually, we have 4 child + 12 grand = 16 centers, but num_fans is 12
        # So we use the 12 grandchild centers as the 12 fans
        self.hierarchy = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        
        # Store child positions for draw_parent_fans
        self.child_positions = self.centers[:4].copy()
        
        # Update centers to only use grandchild positions (12 centers)
        self.centers = self.centers[4:]  # Skip the 4 child nodes, use 12 grandchild nodes
        
        # Store parameters for draw_poincare_gaussian
        self.sigma_child = 0.6
        self.sigma_grand = 0.4
        self.alpha_child_peak = 0.07
        self.alpha_grand_peak = 0.22
        
        # Store base colors for parent nodes
        self.base_rgb = base_rgb
        
        # Update theta for the 12 centers
        self.theta = np.linspace(0.0 - self.margin, 2 * np.pi - self.margin, self.num_fans, endpoint=False)

    def _poincare_circle_euclid(self, p, rho_h):
        """Convert Poincaré circle to Euclidean circle"""
        def rho_to_r(rho): 
            return np.tanh(rho / 2)
        R = rho_to_r(rho_h)
        norm2 = np.dot(p, p)
        denom = 1 - norm2 * R * R
        if denom <= 0: 
            return None, None
        return ((1 - R * R) / denom) * p, ((1 - norm2) * R) / denom

    def _draw_poincare_gaussian(self, ax, p, sigma, color, alpha_peak, layers=50, disk=None):
        """Draw Gaussian via true Poincaré circles"""
        max_rho = 3 * sigma
        for rho in np.linspace(max_rho, 0, layers, endpoint=False):
            alpha = alpha_peak * np.exp(-rho**2 / (2 * sigma**2))
            if alpha < 0.004: 
                continue
            c, r = self._poincare_circle_euclid(p, rho)
            if c is None or r <= 1e-4: 
                continue
            if np.linalg.norm(c) - r >= 1: 
                continue
            circ = Circle(c, r, color=color, alpha=alpha, lw=0)
            if disk is not None:
                circ.set_clip_path(disk)
            ax.add_patch(circ)

    def draw_parent_fans(self, ax):
        """Draw parent and child Gaussians using draw_poincare_gaussian"""
        # Create disk for clipping
        disk = Circle((0, 0), 1, ec='black', fill=False, lw=1.2)
        ax.add_patch(disk)
        ax.set_clip_path(disk)
        
        # Draw parent Gaussians at child positions (4 parent nodes)
        for i in range(4):
            base_color = self.base_rgb[i]
            child_pos = self.child_positions[i]
            # Draw parent Gaussian
            self._draw_poincare_gaussian(ax, child_pos, self.sigma_child, base_color, 
                                         self.alpha_child_peak, disk=disk)
        
        # Draw child Gaussians at grandchild positions (12 child nodes)
        for i in range(4):
            base_color = self.base_rgb[i]
            # Get the 3 grandchild positions for this parent
            for j in range(3):
                grand_idx = i * 3 + j
                grand_pos = self.centers[grand_idx]
                grand_color = self.colors[grand_idx]
                # Draw child Gaussian
                self._draw_poincare_gaussian(ax, grand_pos, self.sigma_grand, grand_color, 
                                           self.alpha_grand_peak, disk=disk)

    def draw_samples(self, ax):
        super().draw_samples(ax, "hyp2")


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
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
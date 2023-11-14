import os
import shutil
import time
import torch
from torchdyn.core import NeuralODE
from torchcfm.model.models import MLP
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher
from utils import torch_wrapper, plot_trajectories, plot_trajectories_cond, HypToyData, EucToyData

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
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4)
FM = ConditionalFlowMatcher(sigma=sigma)
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

# Create colors
color_inx = [4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
cm = mpl.colormaps['tab20b']
colors = cm(np.linspace(0, 1, 20))
colors = colors[color_inx]

# generate data
eucdata = EucToyData()

x0, x1, y0, y1 = eucdata.generate_opposite_samples(batch_size)

# # test: visualize the data in a temp.png
# plt.scatter(x0[:, 0], x0[:, 1], s=10, alpha=0.8, c=colors[y0])
# plt.savefig("temp1.png")
# plt.scatter(x1[:, 0], x1[:, 1], s=10, alpha=0.8, c=colors[y1])
# plt.savefig("temp2.png")
# exit()

start = time.time()
for k in range(10000):
    optimizer.zero_grad()

    x0, x1, y0, y1 = eucdata.generate_opposite_samples(batch_size)

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

        # source_samples, _ = eucdata.get_source_samples(n_samples = 2048)
        source_samples, target_samples, source_labels , target_labels = eucdata.generate_opposite_samples(batch_size)
        source_samples = source_samples.cuda()
        with torch.no_grad():
            traj = node.trajectory(
                source_samples,
                t_span=torch.linspace(0, 1, 100).cuda(),
            )
            # plot_trajectories(traj.cpu().numpy(), k, traj_dir)
            plot_trajectories_cond(traj.cpu().numpy(), source_labels, target_labels, target_samples, k, traj_dir)
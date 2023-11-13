import os
import shutil
import time
import torch
from torchdyn.core import NeuralODE
from torchcfm.model.models import MLP
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from utils import torch_wrapper, plot_trajectories, HypToyData, EucToyData

traj_dir = "trajectory_hyp"
shutil.rmtree(traj_dir, ignore_errors=True)
os.makedirs(traj_dir, exist_ok=True)

# define the model
sigma = 1e-3
dim = 2
batch_size = 512
model = MLP(dim=dim, time_varying=True).cuda()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4)
FM = ConditionalFlowMatcher(sigma=sigma)

# generate data
eucdata = EucToyData()

start = time.time()
for k in range(10000):
    optimizer.zero_grad()

    x0, x1, _, _ = eucdata.generate_opposite_samples(batch_size)

    x0, x1 = x0.cuda(), x1.cuda()
    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

    vt = model(torch.cat([xt, t[:, None]], dim=-1))
    loss = torch.mean((vt - ut) ** 2)

    loss.backward()
    optimizer.step()

    if (k + 1) % 500 == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        start = end
        node = NeuralODE(
            torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )

        source_samples, _ = eucdata.get_source_samples(n_samples = 2048)
        source_samples = source_samples.cuda()
        with torch.no_grad():
            traj = node.trajectory(
                source_samples,
                t_span=torch.linspace(0, 1, 100).cuda(),
            )
            plot_trajectories(traj.cpu().numpy(), k, traj_dir)
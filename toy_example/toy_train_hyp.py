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
sigma = 0.1
dim = 2
batch_size = 256
model = MLP(dim=dim, time_varying=True).cuda()
optimizer = torch.optim.Adam(model.parameters())
FM = ConditionalFlowMatcher(sigma=sigma)

# generate data
hypdata = HypToyData()
source_samples, _ = hypdata.get_source_samples()
target_samples    = hypdata.get_target_samples()

start = time.time()
for k in range(20000):
    optimizer.zero_grad()

    x0, _  = hypdata.get_source_samples(batch_size)
    x1     = hypdata.get_target_samples(batch_size)
    x0, x1 = x0.cuda(), x1.cuda()

    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

    vt = model(torch.cat([xt, t[:, None]], dim=-1))
    loss = torch.mean((vt - ut) ** 2)

    loss.backward()
    optimizer.step()

    if (k + 1) % 5000 == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        start = end
        node = NeuralODE(
            torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )

        source_samples, _ = hypdata.get_source_samples(n_samples = 2048)
        source_samples = source_samples.cuda()
        with torch.no_grad():
            traj = node.trajectory(
                source_samples,
                t_span=torch.linspace(0, 1, 100).cuda(),
            )
            plot_trajectories(traj.cpu().numpy(), k, traj_dir)

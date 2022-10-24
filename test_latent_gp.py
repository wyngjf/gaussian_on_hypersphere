import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import get_mesh_grid


def latent_gaussian_process(time_steps=30, l=2, sig_f=1, sig_n=0.1, device=None, mu=None):
    mesh = get_mesh_grid(time_steps, dim=2, range=[1, time_steps])
    K = np.square(sig_f) * np.exp(- np.square(mesh[:, 0] - mesh[:, 1]) / (2 * np.square(l)))
    K = K.reshape(time_steps, time_steps) + sig_n * torch.eye(time_steps)

    if mu is None:
        mu = torch.zeros([time_steps])
    scale = torch.linalg.cholesky(K)
    mvn = torch.distributions.MultivariateNormal(loc=mu.to(device), scale_tril=scale.to(device))
    return mvn


T = 30
n_trials = 20
lgp = latent_gaussian_process(time_steps=T, device="cpu")
s = lgp.sample((3 * n_trials, ))
color_spec = plt.get_cmap("plasma")
colors = color_spec(np.linspace(0, 1, n_trials))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for i in range(n_trials):
    ax.plot(s[3*i], s[3*i+1], s[3*i+2], color=colors[i])

plt.show()

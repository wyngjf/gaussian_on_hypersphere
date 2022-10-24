import numpy as np
import torch
import matplotlib.pyplot as plt
from von_mises_fisher import VonMisesFisher
from utils import quaternion_average, quaternion_rotate
from utils import set_3d_equal_auto, draw_frame_3d, set_3d_ax_label
from utils import sphere_logarithmic_map, multivariate_normal, get_covariance
import robot_utils
color_spec = plt.get_cmap("plasma")


local_traj = np.load("./local_traj.npy")
batch, dim, time = local_traj.shape
# each quaternion corresponds to a local frame, visualize them by only show the
# tip of the x basis vector
x_trajs = []
for idx_trial in range(batch):
    x_trajs.append(np.array([
        quaternion_rotate(np.array([0, 1., 0., 0.]), local_traj[idx_trial, :, tt]) for tt in range(time)
    ]))
x_trajs = np.array(x_trajs)


def vis_quat_traj_with_x_axis(ax, x_trajs):
    quat_0 = np.array([1.0, 0.0, 0, 0])
    draw_frame_3d(ax, quat_0, scale=4.5)

    colors = color_spec(np.linspace(0, 1, batch))
    for b in range(batch):
        ax.plot(x_trajs[b, :, 1], x_trajs[b, :, 2], x_trajs[b, :, 3], color=colors[b], linestyle="--", alpha=0.8)
    set_3d_ax_label(ax, ["x", "y", "z"])


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
vis_quat_traj_with_x_axis(ax, x_trajs)
plt.draw()


# compute the mean of all quaternions in the demonstrations, and define a dummy vmf around the mean
loc = quaternion_average(torch.from_numpy(local_traj.transpose((0, 2, 1))).reshape(-1, 4))
scale = torch.ones(1) * 25.0
vmf = VonMisesFisher(loc=loc, scale=scale, k=1)

# sampling from vmf and compute the likelihood for visualization
s = vmf.sample(10000)

# lh = torch.exp(vmf.log_prob(s))
# s_color_idx = (lh - lh.min())/(lh.max() - lh.min())
s_x = np.zeros((10000, 4))
for i in range(10000):
    s_x[i] = quaternion_rotate(np.array([0, 1., 0., 0.]), s[i])


vis_sample = True  # True: vis sampled points; False: vis demo traj
use_vmf = True      # True: use vmf to compute likelihood, False: use your code

if vis_sample:
    data = s.numpy()
    viz_point = s_x
else:
    data = local_traj.transpose((0, 2, 1)).reshape(-1, 4)
    viz_point = x_trajs.reshape(-1, 4)
nb_data, nb_dim = data.shape

xts = np.zeros((nb_data, nb_dim))
for n in range(nb_data):
    xts[n, :] = sphere_logarithmic_map(data[n].flatten(), loc.numpy().flatten()).flatten()

if use_vmf:
    prob = torch.exp(vmf.log_prob(torch.from_numpy(data))).numpy()
else:
    covariances = np.identity(4, dtype=float) * 0.1

    # Design of covariance matrix and project it to tangent space
    # covariances = get_covariance(4)
    # w, v = np.linalg.eigh(covariances)
    # ic(w, v)
    # proj_matrix = np.identity(4, dtype=float) - np.einsum("i,j->ij", loc, loc)
    # ic(proj_matrix)
    # covariances = proj_matrix.dot(v)
    # ic(covariances)

    prob = multivariate_normal(xts, np.zeros_like(loc.numpy()), covariances, log=False)

prob_max, prob_min = prob.max(), prob.min()
color_idx = (prob - prob_min) / (prob_max - prob_min)
color = color_spec(color_idx)

# Visualize
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
vis_quat_traj_with_x_axis(ax, x_trajs)
ax.scatter3D(viz_point[:, 1], viz_point[:, 2], viz_point[:, 3], color=color, s=50)
draw_frame_3d(ax, loc, scale=1.5)
set_3d_equal_auto(ax, [-1, 1], [-1, 1], [-1, 1])
plt.show()




import torch
import numpy as np


def track_traj_rets(current_traj_rets, step_rets):
    """
    Update the different returns after a new step was taken in the environment.

    Args:
        current_traj_rets (dict): empty if first step, otherwise depends on prev steps
        step_rets (list): a list of dict, where each list corresponds to an env, and each
            dictionary contains a list of all values from the DMP trajectory.
    """
    sum_step_rets = {}
    for env_idxs, dmp_traj_rets in enumerate(step_rets):
        if isinstance(dmp_traj_rets, dict):  # in case no DMP
            dmp_traj_rets = [dmp_traj_rets]
        for traj_step_rets in dmp_traj_rets:
            for k, v in traj_step_rets.items():
                if k in sum_step_rets:
                    sum_step_rets[k] +=  v
                else:
                    sum_step_rets[k] = v

    for k, v in sum_step_rets.items():
        mean_val = v / (len(step_rets) - 1)
        if k in current_traj_rets:
            current_traj_rets[k] += mean_val
        else:
            current_traj_rets[k] = mean_val
    return current_traj_rets


def write_rewards(writer, traj_rets, traj_ret, epoch, alias, iterations, track=True):
    if track:
        writer.add_scalar(alias + "trajectory/ret/return", traj_ret, epoch)
        for k, v in traj_rets.items():
            v = np.mean(v) if hasattr(v, "__iter__") else v
            writer.add_scalar(alias + "trajectory/ret/" + k, v / iterations, epoch)


def writer_log(
    writer,
    traj_rets,
    traj_ret,
    traj_steps,
    init_pos_heatmap,
    init_pos_heatmap_,
    avg_heatmap,
    epoch,
    alias,
    iterations,
    track_rew=True,
):
    write_rewards(writer, traj_rets, traj_ret, epoch, alias, iterations, track_rew)
    writer.add_scalar(alias + "trajectory/steps", traj_steps, epoch)
    if epoch % 100 == 0:
        h = (init_pos_heatmap / init_pos_heatmap.max()).flatten()
        h[torch.nonzero(h)] = (h[torch.nonzero(h)] + 0.5) / 1.5
        h = torch.stack([h] * 3).reshape((3,) + init_pos_heatmap.shape)
        writer.add_image(alias + "img/init_pos_heatmap_" + str(epoch), torch.abs(h - 1))
        init_pos_heatmap *= 0
        if "train" in alias:
            writer.add_scalar(
                alias + "trajectory/heatmap/mean", avg_heatmap.mean(), epoch
            )
            writer.add_scalar(alias + "trajectory/heatmap/std", avg_heatmap.std(), epoch)
            writer.add_scalar(alias + "trajectory/heatmap/min", avg_heatmap.min(), epoch)
            writer.add_scalar(alias + "trajectory/heatmap/max", avg_heatmap.max(), epoch)
            writer.add_scalar(
                alias + "trajectory/heatmap/percentile", avg_heatmap.quantile(0.95), epoch
            )
            avg_heatmap -= avg_heatmap.min() if avg_heatmap.min() < 0.0 else 0.0
            h = torch.stack([avg_heatmap / avg_heatmap.max()] * 3)
            writer.add_image(alias + "img/avg_heatmap_" + str(epoch), torch.abs(h - 1))
            avg_heatmap *= 0
    if epoch % 10 == 0:
        h = (init_pos_heatmap_ / init_pos_heatmap_.max()).flatten()
        h[torch.nonzero(h)] = (h[torch.nonzero(h)] + 0.5) / 1.5
        h = torch.stack([h] * 3).reshape((3, ) + init_pos_heatmap_.shape)
        writer.add_image(
            alias + "img/init_pos_heatmap_" + str(epoch % 100), torch.abs(h - 1)
        )
        init_pos_heatmap_ *= 0
    return init_pos_heatmap, init_pos_heatmap_, avg_heatmap


def track_actions(writer, actions, current_step, alias):
    writer.add_scalar(alias + "actions/mean", actions.mean(), current_step)
    writer.add_scalar(alias + "actions/std", actions.std(), current_step)
    writer.add_scalar(alias + "actions/min", actions.min(), current_step)
    writer.add_scalar(alias + "actions/max", actions.max(), current_step)


def track_actions_std(writer, stds, current_step, alias):
    writer.add_scalar(alias + "stds/mean", stds.mean(), current_step)


def track_module_weights(writer, model, current_step):
    weight_means, weight_stds, weight_maxs, weight_mins = [], [], [], []
    for p in model.parameters():
        weight_means.append(p.cpu().detach().numpy().mean())
        weight_stds.append(p.cpu().detach().numpy().std())
        weight_mins.append(p.cpu().detach().numpy().min())
        weight_maxs.append(p.cpu().detach().numpy().max())
    writer.add_scalar("parameters/mean", np.array(weight_means).mean(), current_step)
    writer.add_scalar("parameters/std", np.array(weight_stds).mean(), current_step)
    writer.add_scalar("parameters/min", np.array(weight_mins).mean(), current_step)
    writer.add_scalar("parameters/max", np.array(weight_maxs).mean(), current_step)

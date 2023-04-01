import torch
import torch.nn.functional as F
import numpy as np

from algos.base import BaseRLAlgo
from utils import plot, rl_func, indexing, track


class VPGAlgo(BaseRLAlgo):
    def __init__(self, config, experiment_dir, buffer_dir, test):
        """
        VPG requires to work with padded observations since the orientation is encoded
        directly in the observation. This also changes the observations size used for
        critic and buffer.
        """
        self.obs_size = config["training"]["obs_size"]  # width = height of img obs
        self.padding = int(np.ceil(np.ceil(self.obs_size * (np.sqrt(2) - 1)) / 2))
        self.critic_obs_size = self.obs_size + self.padding * 2
        self.n_orient = config["algo"]["n_orientations"]
        self.map_opposite_ret = {
            "push": lambda r: 0.0 if r == 0.5 else 0.5,
            "grasp": lambda r: 0.0 if r == 1.0 else 1.0
        }
        self.critic_name = "VPGCritic"

        super().__init__(config, experiment_dir, buffer_dir, test)

    def get_rgbd_obs(self):
        """
        After calling the general observation add padding to it to prepare for rotation.

        Return:
            (torch.tensor): scaled down and padded colour obs of size (b, c, w_, h_)
            (torch.tensor): scaled down and padded depth obs of size (b, c, w_, h_)
            (torch.tensor): scaled down colour obs of size (b, c, w, h)
            (torch.tensor): scaled down depth obs of size (b, c, w, h)
        """
        rgb, depth = super().get_rgbd_obs()

        nopad_depth = depth.clone()
        nopad_rgb = rgb.clone()

        # Pad image before rotating, new height and width are same size as diagonal
        rgb = F.pad(rgb, tuple([self.padding] * 4), "constant", 0)
        depth = F.pad(depth, tuple([self.padding] * 4), "constant", 0)

        return rgb, depth, nopad_rgb, nopad_depth

    def process_heatmaps(self, heatmaps):
        """
        Post process heatmaps by removing the added padding in observation preprocessing.

        Args:
            heatmaps (torch.tensor): tensor of heatmaps with shape
                (primitivies, orientations, b, w, h)
        Return:
            (torch.tensor): processed heatmaps shape (primitives, orientations, w, h)
        """
        start_idx, last_idx = self.padding, self.obs_size + self.padding
        heatmaps = heatmaps[:,:,:,start_idx:last_idx,start_idx:last_idx]

        return super().process_heatmaps(heatmaps)

    def sample_batch(self, a_idx, ret):
        """
        As per VPG implementation sample a replay buffer batch (single value actually),
        so that the sample is generated from the same action as the current action taken,
        but with an opposite return resulting from the current action. This makes sense
        for grasping task where only two rewards were given for pushing (0 and 0.5) and
        grasping (0 and 1). In case of multi environment training the return value of the
        first environment is taken as reference.
        """
        query_ret = self.map_opposite_ret[self.actions[a_idx[0]]](ret[0])
        samples = self.buffer.sample_batch(
            self.batch_size,
            method="power",
            reward=query_ret,
            action=a_idx
        )

        return samples

    def update(
        self,
        rgb, depth,
        a_idx, init_pos, orient_idx,
        exp_ret, ret, pred_ret,
        samples=None
    ):
        """
        Update critic. First update the last step executed then run update for a batch of
        experience replay samples.

        Args:
            rgb (torch.Tensor): image tensor of size (batch, c, w, h)
            depth (torch.Tensor): depth tensor of size (batch, c, w, h), all c are equal
            a_idx (np.array): which action to take from the defined primitives
            orient_idx (np.array): which orientation to apply the action
            init_pos (np.array): tuple of pixel coords where action will be taken w/ depth
            exp_ret (np.array): expected returns at position
            ret (np.array): returns after action execution
            pred_ret (np.array): critic expected return at position
        """
        pred_ret = torch.from_numpy(pred_ret).flatten().to(self.device)
        exp_ret = torch.from_numpy(exp_ret).flatten().to(self.device)
        self.set_model_mode("train")
        loss_info = self.backprop(
            rgb, depth, a_idx, init_pos, orient_idx, ret, exp_ret, pred_ret
        )
        loss_info["step_critic_loss"] = loss_info.pop("critic_loss")
        samples = self.sample_batch(a_idx, ret) if samples is None else samples
        loss_info.update(super().update(samples=samples))
        return loss_info

    def traj_step(self, init_pos, a_idx, orient_idx, depth):
        """
        Convert orientation indexes to angles, apply hard coded primitive, get new obs
        after applying the primive, calculate push reward based on depth sensor difference
        between current observation and new observation.
        """
        orientation = orient_idx * 2 * np.pi / self.n_orient
        # execute_action = getattr(self.env, self.actions[a_idx])
        # obs, ret, done, info = execute_action(init_pos, depth, orientation)
        obs, ret, done, info = self.env.step(np.array([self.env.action_space.sample()] * self.num_envs).flatten())

        _, _, _, nopad_depth_ = obs = self.get_rgbd_obs()
        for i, d in enumerate(depth):
            if self.actions[a_idx[i]] == "push":
                ret[i] = rl_func.intrinsic_push_rew(d, nopad_depth_[i])

        return obs, ret, done, info

    def backprop(
        self,
        rgb, depth,
        a_idx, init_pos, orient_idx,
        ret, exp_ret, pred_ret,
        new_pred_ret=False,
        **args,
    ):
        """
        Backpropagation for VPG involves:
            - calculate orientations from indexes
            - grasp action is invariant to a 180 degree rotation
                - two rotation are handled for grasp action
            - make label at init position
            - make auxiliary old_pred at init_pos batch
            - make mask for init position, gradients at other position will be 0
            - duplicate elements in batch because of grasp actions
            - run backprop on critic
            - calculate and return new predictions if necessary
        """
        # create batch of orientations
        orients = [i * 2 * np.pi / self.n_orient for i in orient_idx]
        # use also 180 degree rotataion e.g. 45 -> 225, 270 -> 90 for grasp
        grasp_idx = np.where(np.array(self.actions)[a_idx] == "grasp")[0]
        orients += [(o + np.pi) % (2 * np.pi) for o in np.array(orients)[grasp_idx]]

        mask = self.make_mask(rgb.shape[0], rgb.shape[-2:], init_pos, self.padding)
        mask = torch.cat((mask, mask[grasp_idx]))
        pred_ret = torch.cat((pred_ret, pred_ret[grasp_idx]))
        exp_ret = torch.cat((exp_ret, exp_ret[grasp_idx]))
        rgb = torch.cat((rgb, rgb[grasp_idx]))
        depth = torch.cat((depth, depth[grasp_idx]))
        a_idx = np.concatenate((a_idx, a_idx[grasp_idx]))

        critic_loss_info = super().critic_backprop(
            rgb, depth, a_idx, orients, exp_ret, mask, pred_ret, new_pred_ret
        )

        # Fix pred return by averaging grasping trajectories for both orientations
        if new_pred_ret and len(grasp_idx) > 0:
            new_pred_ret = critic_loss_info["new_pred_ret"]
            new_pred_ret[grasp_idx] += new_pred_ret[-len(grasp_idx):]
            new_pred_ret[grasp_idx] /= 2
            critic_loss_info["new_pred_ret"] = new_pred_ret[:-len(grasp_idx)]
        return critic_loss_info

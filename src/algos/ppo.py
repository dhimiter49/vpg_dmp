import torch
import torch.nn.functional as F
import numpy as np

import nets.rl_nets as nets
from algos.base import BaseRLAlgo
from projections import MAP_TR_LAYER
from utils import plot, rl_func, indexing, track


class PPOAlgo(BaseRLAlgo):
    def __init__(self, config, experiment_dir, buffer_dir, test, critic_name=None):
        """
        Step-based RL algorithm based on REINFORCE as a trajectory generator in the VPG
        framework. Adds to the VPG_DMPAlgo a var traj_steps to control the trajectory
        length and a loop to the main traj_step. At the moment implemented only for
        training policy and VPG VF separately, meaning it is not proper PPO.
        TODO: add Q-function for policy.
        """
        self.obs_size = config["training"]["obs_size"]  # width = height of img obs
        self.traj_steps = config["algo"]["traj_steps"]  # width = height of img obs
        self.critic_obs_size = self.obs_size
        self.n_orient = 1  # only one orientation
        self.critic_name = "PixelPolicy"
        self.policy_cam = config["policy"]["observation_type"]

        super().__init__(config, experiment_dir, buffer_dir, test)

        self.use_tr_layer = config["policy"]["trust_region_layer"]["name"] is not None
        robot_state_dim = len(self.env.robot_state()[0].flatten())\
            if config["policy"]["robot_state_config"]["fusion_method"] else None
        self.use_env_obs = config["policy"]["use_env_obs"]
        obs_space = self.env.observation_space.shape[-1]
        self.policy = nets.GaussianPolicy(
            mlp_config=config["policy"]["mlp_config"],
            obs_dim=3 * self.obs_size ** 2,
            weight_vec_dim=self.action_space,
            backbone_params=config["policy"]["backbone_params"],
            env_obs=obs_space\
                if self.use_env_obs else None,
            robot_state_config=config["policy"]["robot_state_config"],
            robot_state_dim=robot_state_dim,
            device=self.device,
            **config["policy"]["gaussian"],
        )
        self.entropy_coeff = config["policy"]["entropy_coeff"]
        if self.use_tr_layer:
            self.projection = MAP_TR_LAYER[
                config["policy"]["trust_region_layer"]["name"]
            ](
                **config["policy"]["trust_region_layer"]["params"]
            )
        if not test:
            self.policy_opt = getattr(torch.optim, config["policy"]["opt"])(
                self.policy.parameters(),
                **config["policy"]["opt_params"]
            )
            sched = config["policy"]["lr_scheduler"].pop("name")
            self.policy_opt_scheduler = getattr(torch.optim.lr_scheduler, sched)(
                    self.policy_opt, **config["policy"]["lr_scheduler"]
            )
            self.policy_update_freq = config["policy"]["update_freq"]
            self.update_freq = lambda : (
                (self.curr_step + 1) % self.critic_update_freq == 0 or\
                (self.curr_step + 1) % self.policy_update_freq == 0
            )
            self.buffer.policy_traj_len = self.policy_update_freq
        self.policy_update_iter = config["policy"]["update_iter"]
        self.pixel_policy = self.critic
        self.train_critic = config["training"]["train_critic"]
        if not self.testing:
            self.pixel_policy_opt = self.critic_opt
            self.pixel_policy_scheduler = self.critic_opt_scheduler
            self.buffer.init_policy_vars(
                self.action_space,
                algo="vpg_dmp",
                tr_layer=self.use_tr_layer,
                robot_state_dim=robot_state_dim,
                pol_env_obs=obs_space if self.use_env_obs else None,
                traj_steps=self.traj_steps,
            )
        self.update_freq = lambda : (self.curr_step + 1) % self.policy_update_freq == 0

    def get_policy_obs(self, pos=None):
        """
        There are four policy observations implemented which can be set in the config.
        These are based on what camera is used. Excpetion is "rgb_crop" which involves
        using the same observation as the VF and then crop the image around the position
        chosen.
        """
        if self.use_env_obs:
            return torch.from_numpy(self.env.get_obs()).to(self.device, dtype=torch.float)
        rgb = torch.from_numpy(self.env.render(
            mode="rgb_array",
            width=self.obs_size,
            height=self.obs_size,
            camera_name="rgbd" if self.policy_cam == "rgbd_crop" else self.policy_cam
        ).copy())
        rgb = indexing.crop_upsample(rgb, pos, self.obs_size) if pos is not None else rgb
        plot.show_image(rgb[0]) if self.plot else None

        rgb = (rgb / 255).to(self.device, dtype=torch.float)
        rgb = rgb.permute(0, 3, 1, 2)
        return rgb

    def traj_step(self, init_pos, *args):
        """
        In order to run a step with PPO:
            - first find the world position where the suggested pixel is
            - hard set the robot tcp to that position
            - get the policy observation before or after setting the robot to position
              depending on observation chosen
            - run policy using observation
            - save information on action probabilties, policy obs...
            - get next observation
        """
        world_point = self.env.pos_behind_box() if self.optimal_critic\
            else self.env.img_to_world(init_pos)
        if self.plot:
            plot.plot_step_obs(
                self.env.envs[0], self.obs_size, init_pos[0], world_point[0]
            )
        if self.policy_cam == "rgbd_crop":
            pol_obs = self.get_policy_obs(init_pos)
        _, penalty, dist_box_to_tcp_rew = self.env.set_tcp_pos(world_point, hard_set=True)
        if self.policy_cam != "rgbd_crop":
            pol_obs = self.get_policy_obs()

        info_ = {
            "logp": [], "pol_obs": [], "action": [], "robot_state": [], "std":[],
            "reward_info": []
        }
        if self.use_tr_layer:
            info_["mean"] = []
        for _ in range(self.traj_steps):
            robot_state = torch.from_numpy(self.env.robot_state())\
                .to(self.device, dtype=torch.float)
            pol_obs = self.get_policy_obs()
            with torch.no_grad():
                weight_vec, logp_weight_vec, mean, std = self.policy(
                    pol_obs, robot_state, test=self.testing
                )
            obs, ret, done, info = self.env.step(weight_vec.detach().cpu().numpy())
            info_["logp"].append(logp_weight_vec.detach().cpu().numpy())
            info_["pol_obs"].append(pol_obs.cpu().numpy())
            info_["action"].append(weight_vec.detach().cpu().numpy())
            info_["robot_state"].append(robot_state.cpu().numpy())
            if self.use_tr_layer:
                info_["mean"].append(mean.detach().cpu().numpy())
            info_["std"].append(std.detach().cpu().numpy())
            info_["reward_info"].append(list(info["reward_info"]))
        info_["reward_info"].append({
            "critic_penalty": np.array(penalty),
            "critic_box_to_tcp_rew": np.array(dist_box_to_tcp_rew)
        })
        for k, v in info_.items():
            info[k] = np.array(v) if k != "reward_info" else v
        self.env.reset_robot_pos()
        obs = self.get_rgbd_obs()
        ret += 0 if self.critic_name == "PixelPolicy" else penalty
        return obs, ret, done, info

    def sample_batch(self, *args):
        """
        Pass policy sample flag in order to samle only the last trajectory, the batch in
        this case is set to the entire length of the trajectory.
        """
        policy_sample = (self.curr_step + 1) % self.policy_update_freq == 0
        return self.buffer.sample_batch(
            self.policy_update_freq if policy_sample else self.critic_update_freq * 4,
            method=None if policy_sample else "uniform",
            policy_sample=policy_sample
        )

    def compute_policy_loss(
        self, pol_obs, logp_old, old_means, old_stds, actions, adv, robot_states
    ):
        loss, entropy_loss, trust_region_loss =\
            0.0, torch.Tensor([0.0]), torch.Tensor([0.0])
        if self.use_tr_layer:
            _, _, means, stds = self.policy(pol_obs, robot_states)
            proj_p, trust_region_loss, entropy_loss = self.project(
                (means, stds),
                (old_means, old_stds),
            )
            logp = self.policy.log_probability(proj_p, actions)
            loss = rl_func.clip_IS_loss(logp, logp_old, adv, self.clip) +\
                entropy_loss + trust_region_loss
        else:
            _, logp, means, stds = self.policy(pol_obs, robot_states, a=actions)
            loss = rl_func.clip_IS_loss(logp, logp_old, adv, self.clip)
        return loss, logp, entropy_loss, trust_region_loss

    def backprop(
        self,
        rgb, depth,
        a_idx, init_pos, orient_idx,
        ret, ret_, _, pixel_prob_old,
        actions=None, old_means=None, old_stds=None, robot_states=None,
        logp_old=None,
        dones=None,
        pol_obs=None,
        **args
    ):
        pix_pol_losses, pix_pol_ent_losses, pol_losses, pol_ent_losses, pol_tr_losses =\
            [0.0], [0.0], [0.0], [0.0], [0.0]
        if (self.curr_step + 1) % self.policy_update_freq == 0:
            disc_ret_pixel_pos = rl_func.discounted_returns(ret_, dones, self.discount)
            disc_ret = rl_func.discounted_returns(ret, dones, self.discount)
            if self.use_tr_layer and self.projection.initial_entropy is None:
                self.projection.initial_entropy =\
                    self.policy.entropy((old_means, old_stds)).mean()
            for i in range(self.policy_update_iter):
                idxs = np.random.choice(
                    np.arange(self.policy_update_freq * self.num_envs),
                    self.batch_size,
                    replace=False
                )

                if self.train_critic:
                    self.pixel_policy_opt.zero_grad()
                    val_heatmap = self.process_heatmaps(
                        self.pixel_policy(rgb[idxs], depth[idxs])
                    )
                    pixel_prob = rl_func.get_prob_at_pos(val_heatmap, init_pos[idxs])
                    ent_loss = -0.1 * torch.distributions.categorical.Categorical(
                        val_heatmap.flatten()
                    ).entropy()
                    loss = rl_func.clip_IS_loss(
                        torch.log(pixel_prob),
                        torch.log(pixel_prob_old.flatten()[idxs]),
                        disc_ret_pixel_pos[idxs],
                        self.clip,
                    ) + ent_loss
                    loss.backward()
                    self.pixel_policy_opt.step()
                    self.pixel_policy_scheduler.step()
                    pix_pol_losses.append(loss.cpu().detach().numpy())
                    pix_pol_ent_losses.append(ent_loss.cpu().detach().numpy())

                self.policy_opt.zero_grad()
                loss, logp, ent_loss, tr_loss = self.compute_policy_loss(
                    pol_obs[idxs],
                    logp_old[idxs],
                    old_means[idxs] if self.use_tr_layer else None,
                    old_stds[idxs] if self.use_tr_layer else None,
                    actions[idxs],
                    disc_ret[idxs],
                    robot_states[idxs] if len(robot_states) > 0 else None,
                )
                loss.backward()
                self.policy_opt.step()
                self.policy_opt_scheduler.step()
                self.buffer.update_logp(logp.cpu().detach().numpy(), idxs)
                pol_losses.append(loss.cpu().detach().numpy())
                pol_ent_losses.append(ent_loss.cpu().detach().numpy())
                pol_tr_losses.append(tr_loss.cpu().detach().numpy())

        # self.entropy_coeff *= 0.99
        return {
            "critic_loss": float(np.mean(pix_pol_losses)),
            "critic_ent_loss": float(np.mean(pix_pol_ent_losses)),
            "policy_loss": float(np.mean(pol_losses)),
            "policy_loss_std": float(np.std(pol_losses)),
            "policy_loss_min": float(np.min(pol_losses)),
            "policy_loss_max": float(np.max(pol_losses)),
            "policy_ent_loss": float(np.mean(pol_ent_losses)),
            "policy_tr_loss": float(np.mean(pol_tr_losses))
        }

    def project(self, gauss, old_gauss):
        proj_p = self.projection(self.policy, gauss, old_gauss, self.curr_step)
        entropy_loss = -self.entropy_coeff * self.policy.entropy(proj_p).mean()
        trust_region_loss = self.projection.get_trust_region_loss(
            self.policy, gauss, proj_p
        )
        return proj_p, trust_region_loss, entropy_loss

    def set_model_mode(self, mode="train"):
        if mode == "train":
            self.critic.train()
            self.policy.train()
        elif mode == "eval":
            self.critic.eval()
            self.policy.eval()

    def save_model(self, epoch, avg_ret):
        if self.save_model_freq == np.inf:
            return
        if epoch % self.save_model_freq == 0:
            torch.save(self.policy.state_dict(), self.experiment_dir + "/policy.pth")
            torch.save(self.critic.state_dict(), self.experiment_dir + "/critic.pth")
        if avg_ret >= self.max_avg_ret:
            self.max_avg_ret = avg_ret
            torch.save(self.policy.state_dict(), self.experiment_dir + "/policy_best.pth")
            torch.save(self.critic.state_dict(), self.experiment_dir + "/critic_best.pth")

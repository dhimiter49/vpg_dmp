import os
import sys
import fancy_gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from buffer import ReplayBuffer
import nets.rl_nets as nets
import env_mode
from utils import rl_func, indexing, track


class BaseRLAlgo:
    def __init__(self, config, experiment_dir, buffer_dir, test=False):
        """
        Algorithm class takes as input the complete configuration dictionary and defines
        the main training workflow in train function. Initalization is changed depending
        on specific algorithm.

        Args:
            config (dict): dictionary of configuration values
            experiment_dir (str): directory where to save experiment
            buffer_dir (str): directory where the buffer observations are saved
            test (bool): true if only testing and no training
        """
        if test:
            config["env"]["run_mode"] = "sync"
            config["env"]["num_envs"] = 1
            config["training"]["cuda"] = True
        self.writer = SummaryWriter(experiment_dir)
        self.writer.add_text("Configuration", str(config))
        self.epochs = config["training"]["epochs"] if not test else\
            config["training"]["test_iter"]
        self.obs_size = config["training"]["obs_size"]  # width = height of img obs
        self.batch_size = config["training"]["batch_size"]
        self.buffer_size = config["training"]["buffer_size"]
        self.device = torch.device(
            "cuda"
            if (config["training"]["cuda"] and torch.cuda.is_available())
            else "cpu"
        )

        self.push_only = config["algo"]["push_only"]
        self.grasp_only = config["algo"]["grasp_only"]
        self.num_envs = config["env"]["num_envs"]
        self.plot = config["env"]["plot"]
        seeds = np.random.choice(list(range(1000)), self.num_envs, replace=False)
        torch.set_num_threads(1)
        if config["env"]["run_mode"] == "SLAsync":
            env_fns = [env_mode.make_slasync_env(
                config["env"]["name"],
                seed=int(seeds[i]),
                width=self.obs_size,
                height=self.obs_size,
            ) for i in range(self.num_envs)]
            self.env = getattr(
                env_mode, config["env"]["run_mode"] + "BoxPushingBinEnv"
            )(env_fns)
        else:
            env_fns = [lambda: fancy_gym.make(
                config["env"]["name"],
                seed=int(seeds[i]),
                width=self.obs_size,
                height=self.obs_size,
            ) for i in range(self.num_envs)]
            self.env = getattr(
                env_mode, config["env"]["run_mode"].capitalize() + "BoxPushingBinEnv"
            )(env_fns)
        torch.set_num_threads(os.cpu_count())
        self.action_space = self.env.action_space.shape[-1]
        self.pos_neigh = config["training"]["position_neighbourhood"]
        self.gauss_filter = nets.utils.create_gauss_filter(self.pos_neigh, self.device)
        self.critic = getattr(nets, self.critic_name)(
            n_orientations=self.n_orient,
            obs_size=self.critic_obs_size,
            push_only=self.push_only,
            grasp_only=self.grasp_only,
            backbone_params=config["training"]["backbone_params"],
            device=self.device,
        )

        if not test:
            self.buffer = ReplayBuffer(
                self.buffer_size,
                self.critic_obs_size,
                buffer_dir,
                self.num_envs,
                self.pos_neigh,
            )
            self.critic_opt = getattr(torch.optim, config["training"]["critic_opt"])(
                self.critic.parameters(),
                **config["training"]["critic_opt_params"]
            )
            sched = config["training"]["lr_scheduler"].pop("name")
            self.critic_opt_scheduler = getattr(torch.optim.lr_scheduler, sched)(
                    self.critic_opt, **config["training"]["lr_scheduler"]
            )
            self.critic_loss = getattr(torch.nn, config["training"]["loss"])(reduce=False)
            self.adv_loss = config["training"]["adv_loss"]
            self.critic_loss.to(device=self.device)
            self.critic_update_freq = config["training"]["update_freq"]
        self.curr_step = 0
        self.update_freq = lambda: (self.curr_step + 1) % self.critic_update_freq == 0

        actions = []
        if not self.grasp_only:
            actions.append("push")
        if not self.push_only:
            actions.append("grasp")
        self.actions = np.array(actions)
        self.eps = config["algo"]["epsilon"]
        self.init_eps = config["algo"]["epsilon"]
        self.eps_decay = config["algo"]["epsilon_decay"]
        self.discount = config["algo"]["discount"]
        self._lambda = config["algo"]["lambda"]
        self.clip = config["algo"]["clip"]

        # Number of primitives conditions the algorithm
        self.n_primitives = (not self.grasp_only) + (not self.push_only)
        self.eps_decay = self.eps_decay and self.n_primitives > 1
        self.eps = 0 if self.n_primitives == 1 else self.eps

        self.experiment_dir = experiment_dir
        self.testing = test
        self.init_pos_heatmap = torch.zeros((self.obs_size, self.obs_size))
        self.init_pos_heatmap_ = torch.zeros((self.obs_size, self.obs_size))
        self.avg_heatmap = torch.zeros((self.obs_size, self.obs_size))
        self.max_avg_ret = -np.inf
        self.save_model_freq = config["training"]["save_model_freq"]
        self.test_freq = config["training"]["test_freq"]
        self.test_iter = config["training"]["test_iter"]
        self.fix_position = config["training"]["fix_position"]
        self.optimal_critic = config["training"]["optimal_critic"]

    def loop(self, epochs, init_idx=0, test=False):
        """
        Main loop can be used for training and testing. The loop ends after the number of
        specified epochs have been completed. Each epoch is a complete trajectory that
        either ends because of a done flag from the environment or the maximum number of
        trajectory steps has been exceeded.

        In the loop we first read the current observation by rendering the rgb and the
        depth information from the camera. Based on this observation our critic returns
        a heatmap of the image that corresponds to the expected reward.

        Args:
            epochs (int): number of epcohs to run the loop for
            init_idx (int): initial index to start from, necessary for testing
            test (bool): specifies to run loop in test mode
        """
        epoch, traj_begin, epochs = init_idx, self.curr_step, init_idx + epochs
        traj_ret, traj_rets, done = np.zeros(self.num_envs), {}, np.ones(self.num_envs)
        pbar = tqdm(total=epochs - epoch, leave=False)
        alias = "test/" if test else "train/"
        _, obs = self.env.reset(), self.get_rgbd_obs()
        while epoch < epochs:
            # Take a step in the env
            rgb, depth = obs[:2]
            with torch.no_grad():
                val_heatmap = self.process_heatmaps(self.critic(rgb, depth))

            init_pos, a_idx, orient_idx, pred_ret = self.get_action(val_heatmap, obs[-1])
            obs_, ret, done, info = self.traj_step(init_pos, a_idx, orient_idx, obs[-1])
            next_pred_ret = self.get_next_pred(obs_)
            exp_ret = ret + self.discount * next_pred_ret
            pixel_pos_ret = rl_func.pixel_pos_ret(info["reward_info"])  # pixel-policy ret

            # Update for next iteration
            if not test:
                self.buffer.store(
                    rgb.cpu().numpy(), depth.cpu().numpy(),
                    a_idx, init_pos, orient_idx,
                    ret, exp_ret, pred_ret, pixel_pos_ret,
                    done, info
                )

                if self.update_freq():
                    loss = self.update(
                        rgb, depth, a_idx, init_pos, orient_idx, exp_ret, ret, pred_ret,
                        samples=None
                    )
                    print("\t".join(f"{k}: {v}" for k, v in loss.items()))
                    for key in loss:
                        if loss[key] != 0.0:
                            alias_ = key.replace("_loss", "")
                            self.writer.add_scalar(
                                "loss/" + alias_, loss[key], self.curr_step
                            )

            obs, self.curr_step, traj_ret = obs_, self.curr_step + 1, traj_ret + ret
            traj_rets = track.track_traj_rets(traj_rets, info["reward_info"])
            track.track_actions(self.writer, info["action"], self.curr_step, alias)
            track.track_actions_std(self.writer, info["std"], self.curr_step, alias)
            if not test:
                track.track_module_weights(self.writer, self.policy, self.curr_step)

            if done.sum() > 0:  # one of the env is over, reset all
                if test:
                    self.save_model(epoch, traj_ret.mean())
                (
                    self.init_pos_heatmap,
                    self.init_pos_heatmap_,
                    self.avg_heatmap
                ) = track.writer_log(
                    self.writer,
                    traj_rets,
                    traj_ret.mean() if not test else traj_ret.sum() / self.test_iter,
                    (self.curr_step - traj_begin if not test else
                        (self.curr_step - traj_begin) * self.num_envs / self.test_iter),
                    self.init_pos_heatmap, self.init_pos_heatmap_, self.avg_heatmap,
                    epoch if not test else init_idx,
                    alias, -(self.test_iter // -self.num_envs) if test else 1,
                    not test or epoch == epochs - self.num_envs,
                )
                self.env.reset()
                if not test and epoch % self.test_freq == 0:
                    self.test_training(epoch)  # test while training
                pbar.update(self.num_envs)
                traj_begin = self.curr_step if not test else traj_begin
                traj_ret = np.zeros(self.num_envs) if not test else traj_ret
                traj_rets = {} if not test else traj_rets
                obs = self.get_rgbd_obs()  # rgb, depth, (nopad_rgb, nopad_depth)
                if epoch == 0:
                    self.writer.add_image("img/obs", obs[0][0])
                epoch += self.num_envs
                if self.eps_decay:
                    self.eps = max(self.init_eps * np.power(0.9998, self.epochs), 0.1)
        pbar.close()

    def test_training(self, epoch):
        """
        Test during training. In this case some class variables are overwrittne before
        starting a test loop and afterwards as well.
        """
        self.testing, curr_step = True, self.curr_step
        h_, h__ = self.init_pos_heatmap, self.init_pos_heatmap_
        self.init_pos_heatmap *= 0
        self.init_pos_heatmap_ *= 0
        self.loop(epochs=self.test_iter, init_idx=epoch, test=True)
        self.init_pos_heatmap, self.init_pos_heatmap_ = h_, h__
        self.testing, self.curr_step = False, curr_step

    def train(self, critic_path, policy_path):
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        if policy_path is not None:
            self.policy.load_state_dict(torch.load(policy_path))
        self.loop(epochs=self.epochs)
        self.env.close()

    def test(self):
        """
        Testing runs training in evaluation mode. Update frequency is set to infinity in
        order to not update models anymore. Load models and run training (buffer is not
        initialized).
        """
        self.update_freq = lambda: False
        self.save_model_freq = np.inf
        self.policy.load_state_dict(
            torch.load(
                self.experiment_dir + "policy_best.pth",
                map_location=self.device
            )
        )
        self.critic.load_state_dict(
            torch.load(
                self.experiment_dir + "critic_best.pth",
                map_location=self.device
            )
        )
        self.policy.eval()
        self.critic.eval()
        self.loop(epochs=self.epochs, test=True)
        self.env.close()

    def get_rgbd_obs(self):
        """
        Returns the current environment state. The RGB-D camera is used to generate the
        observation.

        Return:
            (torch.tensor): scaled down and padded colour obs of size (b, c, w, h)
            (torch.tensor): scaled down and padded depth obs of size (b, c, w, h)
        """
        rgb = torch.from_numpy(self.env.render(
            mode="rgb_array",
            width=self.obs_size,
            height=self.obs_size,
            camera_name="rgbd"
        ).copy())
        depth = torch.from_numpy(self.env.render(
            mode="depth_array",
            width=self.obs_size,
            height=self.obs_size,
            camera_name="rgbd"
        ).copy())

        # Scale rgb, add batch dim, set type and device
        rgb = (rgb / 255).to(self.device, dtype=torch.float)
        depth = (depth.unsqueeze(-1)).to(self.device, dtype=torch.float)

        # (batch, channels, w, h)
        rgb = rgb.permute(0, 3, 1, 2)
        depth = depth.permute(0, 3, 1, 2)

        return rgb, depth

    def process_heatmaps(self, heatmaps):
        """
        Args:
            heatmaps (torch.tensor): tensor of heatmaps with shape
                (primitivies, orients, batch, w, h)
        Return:
            (torch.tensor): processed heatmaps shape (batch, primitives, orients, w, h)
        """
        if torch.isnan(heatmaps).sum() > 0.0:
            sys.exit("Critic returned NAN value(s)")
        return heatmaps.permute(2, 0, 1, 3, 4)

    def get_action(self, val_heatmap, depth):
        """
        Compute the initial position of the end effector and the action to carry out from
        the given heatmaps. The input is over multiple environments(batch dim), the output
        will also return an array of values for each environment.

        Args:
            heatmaps (torch.tensor): tensor of heatmaps with size
                (batch, primitivies, orientations, width, height), batch corresponds to
                number of environments running in parallel
            depth (torch.tesnor): depth obs tensor corresponding to heatmaps in w and h
                (batch, 1, width, height), channel dim is 1
        Return:
            (np.array): pixel coords where action will be taken w/ depth value
            (np.array): actions to take from the defined primitives, irrelevant for DMP
            (np.array): orientations to apply the action, irrelevant for DMP case
            (np.array): predicted return of the critic at the sampled pixel
        """
        assert val_heatmap.shape[-2:] == depth.shape[-2:], \
            "Padding and upsampling not compatible!"

        pred_ret, pos, a_idx, orient_idx = [], [], [], []
        if self.testing:
            sample_mode = "max"
        elif self.critic_name == "PixelPolicy":
            sample_mode = "categorical"
        else:
            sample_mode = "softmax_categorical"
        for h in val_heatmap:
            if self.fix_position is not None:
                noise = self.obs_size // 25
                p = tuple(self.fix_position + np.random.randint(-noise, noise, 2))
                # p = tuple(self.fix_position)
                o, idxs = 0, [0]
            else:
                idxs = indexing.unravel_index(rl_func.critic_decision(h, sample_mode), h)
                p, o = idxs[-2:], idxs[1]
            p = indexing.move_from_border(p, self.pos_neigh // 2, self.obs_size)
            p += (depth[0, 0, p[0], p[1]].item(),)  # add depth to pos
            u, d, l, r = indexing.get_neigh(p, self.pos_neigh , self.obs_size)
            pred_ret.append(
                h[:, :, u:d, l:r].detach().cpu().numpy().reshape(
                    self.pos_neigh, self.pos_neigh
                )
            )
            pos.append(p)
            orient_idx.append(o)
            if np.random.uniform() > self.eps:
                a_idx.append(idxs[0])
            else:
                a_idx.append(np.random.randint(0, self.n_primitives))

        for p in pos:
            self.init_pos_heatmap[p[0], p[1]] += 1
            self.init_pos_heatmap_[p[0], p[1]] += 1
        if not self.testing:  # Save for tracking average haetmpas
            self.avg_heatmap += val_heatmap.detach().cpu().sum(0).squeeze()
        return np.array(pos), np.array(a_idx), np.array(orient_idx), np.array(pred_ret)

    def get_next_pred(self, obs):
        with torch.no_grad():
            pred = self.critic(obs[0], obs[1])
        return pred.max().item()

    def traj_step(self, init_pos, a_idx, orient_idx, depth):
        """
        Execute either fixed primitives with hard coded implementation or use DMPs.

        Args:
            init_pos (np.array): array of pixel coords where action will be taken
            a_idx (np.array): which actions to take from the defined primitives
            orient_idx (np.array): which orientation to apply the action
            depth (torch.Tensor): depth tensor of size (batch, c, w, h), all c are equal
        Return:
            (tuple): tuple of tensors containing rgb and depth both padded and not
            (np.array): returns after executing trajectory
            (np.array): if episode has finished during the trajectory
            (dict): dict of more information on trajectory execution
        """
        pass

    def sample_batch(self, *args):
        """
        Per default sample batch randomly.
        """
        return self.buffer.sample_batch(self.batch_size)

    def update(self, *args, samples=None):
        """
        Update from experience, either randomly from buffer or only the last trjectory
        por on-policy update.
        """
        samples = self.sample_batch() if samples is None else samples
        loss_replay_info = {}
        if len(samples) > 0:
            loss_replay_info = self.backprop(
                rgb=torch.from_numpy(samples["rgb"]).to(self.device),
                depth=torch.from_numpy(samples["depth"]).to(self.device),
                a_idx=samples["a_idx"],
                init_pos=samples["init_pos"],
                orient_idx=samples["orient_idx"],
                ret=torch.from_numpy(samples["ret"]).to(self.device),
                ret_=torch.from_numpy(samples["pixel_pos_ret"]).to(self.device),
                exp_ret=torch.from_numpy(samples["exp_ret"]).to(self.device),
                pred_ret=torch.from_numpy(samples["pred_ret"]).to(self.device),
                actions=torch.from_numpy(samples["actions"]).to(self.device),
                old_means=torch.from_numpy(samples["means"]).to(self.device),
                old_stds=torch.from_numpy(samples["stds"]).to(self.device),
                logp_old=torch.from_numpy(samples["logps"]).to(self.device),
                robot_states=torch.from_numpy(samples["robot_states"]).to(self.device),
                dones=torch.from_numpy(samples["dones"]).to(self.device, dtype=int),
                pol_obs=torch.from_numpy(samples["pol_obs"]).to(self.device),
                new_pred_ret=True,
            )

            # Handle pred_ret update, depending if vpg or vpg_dmp
            if "new_pred_ret" in loss_replay_info:
                new_pred_ret = loss_replay_info.pop("new_pred_ret")
                ret = samples["ret"].flatten()[:len(new_pred_ret)]
                next_exp_ret = ret + self.discount * new_pred_ret.sum((-1, -2))
                self.buffer.update_pred_ret(new_pred_ret, next_exp_ret)

        return loss_replay_info

    def critic_backprop(
        self,
        rgb, depth,
        a_idx, orients,
        label, mask, old_pred,
        new_pred_ret
    ):
        """
        Backpropagate.

        Args:
            rgb (torch.Tensor): image tensor of size (batch, c, w, h)
            depth (torch.Tensor): depth tensor of size (batch, c, w, h), all c are equal
            a_idx (np.array): action to take from the defined primitives (batch, 1)
            orients (np.array): orientation to apply the action (batch, 1)
            label (torch.Tensor): tensor with expected ret at init_pos, everywhere else 0
            mask (torch.Tensor): tensor with 1 only at init_pos, everywhere else 0
            old_pred (torch.Tensor): old VF predictions are used for clipping
            new_pred_ret (bool): if newly predicted returns should be returned
        """
        self.critic_opt.zero_grad()
        val_heatmaps = self.critic(rgb, depth, orients, True)
        val_heatmaps = val_heatmaps.view((-1,) + val_heatmaps.shape[-2:])
        a_batch_idxs = torch.arange(rgb.shape[0]) + rgb.shape[0] * a_idx
        pred = val_heatmaps[a_batch_idxs].to(self.device) * mask
        loss = rl_func.critic_clip_loss(
            pred.sum((-1, -2)), old_pred.sum(-1), label, self.clip, self.critic_loss
        )
        loss.backward()
        self.critic_opt.step()
        self.critic_opt_scheduler.step()

        loss_info = {"critic_loss": float(loss)}
        if new_pred_ret:
            pred_ = pred.flatten()
            pred_ = pred_[torch.nonzero(pred_)]
            pred_ = pred_.reshape(pred.shape[0], self.pos_neigh, self.pos_neigh)
            loss_info["new_pred_ret"] = pred_.cpu().detach().numpy()
        return loss_info

    def make_mask(self, batch, img_size, init_pos):
        """
        Create a mask to extract the pixel positions where the Q-function sampled.

        Args:
            batch (int): batch size
            img_size (tuple): tuple of two values for image width and height
            init_pos (tuple): tuple of the positions slected in image

        Return:
            (torch.Tensor): torch tensor with shape (batch,) + img_size
        """
        mask = torch.zeros(batch * img_size[0] * img_size[1]).to(self.device)
        flat_init_pos = (init_pos[:, 0] * img_size[0] + init_pos[:, 1]) +\
            (np.arange(init_pos.shape[0]) * img_size[0] * img_size[1])  # batch prefix
        mask[flat_init_pos] = 1
        mask = mask.reshape((batch,) + img_size)
        if self.pos_neigh > 1:
            mask = self.gauss_filter(mask.unsqueeze(1)).squeeze(1)
            mask[mask < 1e-5] = 0
        return mask

    def save_model(self, epoch, avg_ret):
        if self.save_model_freq == np.inf:
            return
        if epoch % self.save_model_freq == 0:
            torch.save(self.critic.state_dict(), self.experiment_dir + "/critic.pth")
        if avg_ret >= self.max_avg_ret:
            self.max_avg_ret = avg_ret
            torch.save(self.critic.state_dict(), self.experiment_dir + "/critic_best.pth")

    def set_model_mode(self, mode="train"):
        if mode == "train":
            self.critic.train()
        elif mode == "eval":
            self.critic.eval()

import numpy as np
import itertools


class ReplayBuffer:
    def __init__(
        self,
        size: int,
        obs_size: int,
        buffer_dir: str,
        num_envs: int = 1,
        pos_neigh: int = 1,
    ):
        """
        Simple experience replay buffer with FIFO update strategy and random sampling
        strategy.

        Args:
            size (int): buffer size
            obs_size (int): observation size across x and y axis
            buffer_dir (str): directory where to save observations
            num_envs (int): number of environments running in parallel
        """
        self.buffer_dir = buffer_dir
        self.obs_size = obs_size
        self.num_envs = num_envs
        self.rgb_pol_obs = True
        self.algo = "vpg"
        self.action_idxs = np.empty((size, self.num_envs), dtype=int)
        self.rewards = np.empty((size, self.num_envs), dtype=np.float32)
        self.pixel_pos_rewards = np.empty((size, self.num_envs), dtype=np.float32)
        self.expected_rewards = np.empty((size, self.num_envs), dtype=np.float32)
        self.pred_rewards = np.empty(
            (size, self.num_envs, pos_neigh, pos_neigh), dtype=np.float32
        )
        self.init_positions = np.empty((size, self.num_envs, 3), dtype=float)
        self.orientations = np.empty((size, self.num_envs), dtype=int)
        self.dones = np.empty((size, self.num_envs), dtype=bool)
        self.file_name = lambda alias, i: self.buffer_dir + alias + "_" + str(i) + ".npy"
        self.traj_len = 1000  # will be overwritten by the policy
        self.traj_steps = 1
        self.ptr, self.size, self.current_size, self.last_sampled_idxs = 0, size, 0, []

    def init_policy_vars(
        self,
        action_space,
        algo="vpg_dmp",
        tr_layer=False,
        robot_state_dim=None,
        pol_env_obs=None,
        traj_steps=1,
    ):
        self.algo = algo
        self.traj_steps = traj_steps
        if self.algo == "vpg_policy_dmp":
            self.__dict__.pop("expected_rewards", None)
        self.__dict__.pop("orientations", None)
        self.logps = np.empty((self.size, self.num_envs), dtype=float)
        self.actions = np.empty(
            (self.size, self.num_envs, action_space), dtype=np.float32
        )
        if tr_layer:
            self.means = np.empty(
                (self.size, self.num_envs, action_space), dtype=np.float32
            )
            self.stds = np.empty(
                (self.size, self.num_envs, action_space, action_space), dtype=np.float32
            )
        if robot_state_dim is not None:
            self.robot_states = np.empty(
                (self.size, self.num_envs, robot_state_dim), dtype=np.float32
            )
        if pol_env_obs is not None:
            self.rgb_pol_obs = False
            self.pol_obs = np.empty(
                (self.size, self.num_envs, pol_env_obs), dtype=np.float32
            )

    def store(self, rgb_obs, depth_obs, a, p, o, r, r_, r__, r___, d, info=None):
        assert rgb_obs.shape[1:] == (3, self.obs_size, self.obs_size)
        assert depth_obs.shape[1:] == (1, self.obs_size, self.obs_size)

        np.save(self.file_name("rgb", self.ptr), rgb_obs)
        np.save(self.file_name("depth", self.ptr), depth_obs)
        self.action_idxs[self.ptr] = a
        self.rewards[self.ptr:self.ptr + self.traj_steps] = r
        self.pred_rewards[self.ptr] = r__
        self.pixel_pos_rewards[self.ptr] = r___
        self.init_positions[self.ptr] = np.array(p)
        self.dones[self.ptr:self.ptr + self.traj_steps] = d
        if self.algo != "vpg":
            if self.rgb_pol_obs:
                if self.traj_steps == 1:
                    np.save(self.file_name("pol_obs", self.ptr), info["pol_obs"])
                else:
                    for i in range(self.traj_steps):
                        np.save(
                            self.file_name("pol_obs", self.ptr + i), info["pol_obs"][i]
                        )
            else:
                self.pol_obs[self.ptr:self.ptr + self.traj_steps] = info["pol_obs"]
            self.logps[self.ptr:self.ptr + self.traj_steps] = info["logp"]
            self.actions[self.ptr:self.ptr + self.traj_steps] = info["action"]
            if "mean" in info:
                self.means[self.ptr:self.ptr + self.traj_steps] = info["mean"]
                self.stds[self.ptr:self.ptr + self.traj_steps] = info["std"]
        if self.algo == "vpg":
            self.orientations[self.ptr] = o
        if self.algo != "vpg_policy_dmp":
            self.expected_rewards[self.ptr] = r_
        if hasattr(self, 'robot_states'):
            self.robot_states[self.ptr:self.ptr + self.traj_steps] = info["robot_state"]
        self.ptr = (self.ptr + self.traj_steps) % self.size
        self.current_size = min(self.current_size + 1, self.size)

    def sample_batch(
        self,
        batch_size: int = 1,
        method: str = "uniform",
        policy_sample: bool = False,
        reward: float = None,
        action: int = None,
    ):
        """
        Sample batch of transitions (trajectories), condition batch on reward or actions
        based on vpg implementation.

        Args:
            batch_size (int): size of the batch to sample
            method (str): method how to sample distributions
            reward (float): request transitions only with this reward value
            action (int): request transitions only with this action
            policy_sample (bool): sample to update policy, need to return next obs
        Return:
            (tuple): tuple of rgb values, depth, actions, rewards and done flags
        """
        if self.current_size == 0:
            return {}

        valid_idxs = np.arange(self.current_size)
        if policy_sample:
            valid_idxs = np.arange(
                self.ptr - self.policy_traj_len * self.traj_steps, self.ptr
            ) % self.size

        if reward is not None:
            valid_idxs = np.intersect1d(valid_idxs, np.where(self.rewards == reward)[0])
        if action is not None:
            valid_idxs = np.intersect1d(
                valid_idxs, np.where(self.action_idxs == action)[0])
        if len(valid_idxs) == 0:
            return {}

        if method == "uniform":
            idxs = np.random.choice(valid_idxs, batch_size, replace=not policy_sample)
        elif method == "power":
            idxs = np.argsort(  # sort by difference of expected ret to pred ret
                np.abs(
                    self.pred_rewards[valid_idxs].mean(-1) -\
                    self.expected_rewards[valid_idxs].mean(-1)
                )
            )[(np.random.power(2, batch_size) * len(valid_idxs)).astype(int)]  # sampling
        else:
            idxs = valid_idxs
        if len(idxs) == 0:
            return {}

        # get beginning of trajecotry indexes, relevant for VF update
        traj_idxs = idxs - np.remainder(idxs, self.traj_steps)
        rgb_obs = [np.load(self.file_name("rgb", i)) for i in traj_idxs]
        depth_obs = [np.load(self.file_name("depth", i)) for i in traj_idxs]
        if policy_sample:
            if self.rgb_pol_obs:
                pol_obs = np.concatenate(
                    [np.load(self.file_name("pol_obs", i)) for i in idxs]
                )
            else:
                pol_obs = self.pol_obs[idxs].reshape(-1, *self.pol_obs.shape[-1:])


        self.last_sampled_idxs = idxs
        return dict(
            rgb = np.concatenate(rgb_obs),
            depth = np.concatenate(depth_obs),
            a_idx = self.action_idxs[traj_idxs].flatten(),
            init_pos = self.init_positions[traj_idxs].reshape(-1, 3),  # batch, xyz
            orient_idx = self.orientations[traj_idxs].flatten()\
                if self.algo == "vpg" else np.array([]),
            ret = self.rewards[idxs],
            pixel_pos_ret = self.pixel_pos_rewards[traj_idxs],
            pred_ret = self.pred_rewards[traj_idxs],
            exp_ret = self.expected_rewards[traj_idxs].flatten()\
                if self.algo != "vpg_policy_dmp" else np.array([]),
            dones = self.dones[idxs],
            # leave out last trajectory since this can't be used for updatee
            pol_obs = pol_obs if policy_sample else np.array([]),
            actions = self.actions[idxs].reshape(-1, *self.actions.shape[-1:])\
                if policy_sample else np.array([]),
            means = self.means[idxs].reshape(-1, *self.means.shape[-1:])\
                if hasattr(self, "means") else np.array([]),
            stds = self.stds[idxs].reshape(-1, *self.stds.shape[-2:])\
                if hasattr(self, "stds") else np.array([]),
            logps = self.logps[idxs].flatten() if policy_sample else np.array([]),
            robot_states = self.robot_states[idxs]\
                .reshape(-1, self.robot_states.shape[-1])\
                if hasattr(self, "robot_states") else np.array([]),
        )

    def update_pred_ret(self, new_pred_ret, new_exp_ret):
        """
        Update predicted return from the critic for the sampled idxs.

        Args:
            pred_ret (np.array): array of new predicted return
        """
        new_pred_ret = new_pred_ret.reshape((-1,) + self.pred_rewards.shape[1:])
        new_exp_ret = new_exp_ret.reshape(-1, self.num_envs)

        assert new_pred_ret.shape == self.pred_rewards[self.last_sampled_idxs].shape or\
            new_pred_ret.shape == self.pred_rewards[self.last_sampled_idxs[:-1]].shape
        assert new_exp_ret.shape == self.expected_rewards[self.last_sampled_idxs].shape\
          or new_exp_ret.shape == self.expected_rewards[self.last_sampled_idxs[:-1]].shape
        self.last_sampled_idxs = self.last_sampled_idxs[:len(new_exp_ret)]
        self.pred_rewards[self.last_sampled_idxs] = new_pred_ret
        self.expected_rewards[self.last_sampled_idxs] = new_exp_ret

    def update_logp(self, new_logps, idxs):
        """
        Update policy action log probabilities.

        Args:
            new_logps (np.array): array of new policy log probabilities
        """
        assert len(new_logps) == len(idxs)
        assert len(self.last_sampled_idxs) * self.num_envs > len(idxs)
        old_logps = self.logps[self.last_sampled_idxs].flatten()
        old_logps[idxs] = new_logps
        self.logps[self.last_sampled_idxs] =\
            old_logps.reshape(len(self.last_sampled_idxs), self.num_envs)

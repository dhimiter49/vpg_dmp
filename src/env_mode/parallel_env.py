import os
import numpy as np
import fancy_gym
from concurrent.futures import ThreadPoolExecutor


def worker(args, env, cmd):
    if cmd == "step":
       return env.step(args)
    elif cmd == "reset":
        return env.reset()
    elif cmd == "reset_mp":
        return env.reset_mp()
    elif cmd == "close":
        return env.close()
    elif cmd == "img_to_world":
        return env.img_to_world(**args)
    elif cmd == "num_boxes_in":
        return env.num_boxes_in()
    elif cmd == "set_tcp_pos":
        return env.set_tcp_pos(**args)
    elif cmd == "reset_robot_pos":
        return env.reset_robot_pos()
    elif cmd == "pos_behind_box":
        return env.pos_behind_box(**args)
    elif cmd == "robot_state":
        return env.robot_state()
    elif cmd == "get_obs":
        return env.get_obs()


class ParallelBoxPushingBinEnv:
    def __init__(self, env_fns, seed=0, **env_kwargs):

        self.num_envs = len(env_fns)
        self.num_workers = min(self.num_envs, os.cpu_count() - 1)
        self.envs = [env_fn() for env_fn in env_fns]
        self.action_space = self.envs[0].action_space
        self.action_space_len = self.envs[0].action_space.shape[0]
        self.observation_space = self.envs[0].observation_space
        self.execute = ThreadPoolExecutor(self.num_workers)

        self.step_cmds = ["step"] * self.num_envs
        self.reset_cmds = ["reset"] * self.num_envs
        self.reset_mp_cmds = ["reset_mp"] * self.num_envs
        self.close_cmds = ["close"] * self.num_envs
        self.img_to_world_cmds = ["img_to_world"] * self.num_envs
        self.set_tcp_pos_cmds = ["set_tcp_pos"] * self.num_envs
        self.reset_robot_cmds = ["reset_robot_pos"] * self.num_envs
        self.pos_behind_box_cmds = ["pos_behind_box"] * self.num_envs
        self.num_boxes_in_cmds = ["num_boxes_in"] * self.num_envs
        self.robot_state_cmds = ["robot_state"] * self.num_envs
        self.get_obs_cmds = ["get_obs"] * self.num_envs
        self.blank = [None] * self.num_envs

    def step(self, actions):
        out = self.execute.map(worker, actions, self.envs, self.step_cmds)
        out = list(out)
        obs = np.empty((self.num_envs, out[0][0].shape[0]))
        ret, done, info = [], [], {}
        info_keys = [key for key in out[0][3].keys()]
        for k in info_keys:
            info[k] = []
        for i, o in enumerate(out):
            obs[i] = np.array(o[0])
            ret.append(o[1])
            done.append(o[2])
            for k in info_keys:
                info[k].append(o[3][k])
        return obs, np.array(ret), np.array(done), info

    def render(self, **args):
        """
        Parallelization not possible.
        Args:
            args: list of dict containing: mode, width, height, camera_name
        """
        return np.array([self.envs[i].render(**args) for i in range(self.num_envs)])

    def reset(self):
        out = self.execute.map(worker, self.blank, self.envs, self.reset_cmds)
        return np.array(list(out))

    def reset_mp(self):
        out = self.execute.map(worker, self.blank, self.envs, self.reset_mp_cmds)
        return np.array(list(out))

    def close(self):
        out = self.execute.map(worker, self.blank, self.envs, self.close_cmds)

    def img_to_world(self, pixel_pos, cam=None):
        """
        Args:
            pixel_pos (list): list of pixel positions for each env
            cam (str): camera
        """
        args = [{"pixel_pos": p} for p in pixel_pos]
        if cam is not None:
            for a in args: a["cam"] = cam
        out = self.execute.map(worker, args, self.envs, self.img_to_world_cmds)
        return list(out)

    def num_boxes_in(self):
        out = self.execute.map(worker, self.blank, self.envs, self.num_boxes_in_cmds)
        return np.array(list(out))

    def robot_state(self):
        out = self.execute.map(worker, self.blank, self.envs, self.robot_state_cmds)
        return np.array(list(out))

    def get_obs(self):
        out = self.execute.map(worker, self.blank, self.envs, self.get_obs_cmds)
        return np.array(list(out))

    def set_tcp_pos(self, pos, hard_set=None):
        """
        Args:
            pos (list): list of simulation positions to move the tcp to
            hard_set (bool): hard set robot to assigned position
        """
        args = [{"desired_tcp_pos": p} for p in pos]
        if hard_set is not None:
            for a in args: a["hard_set"] = hard_set
        out = self.execute.map(worker, args, self.envs, self.set_tcp_pos_cmds)
        out = list(out)
        positions, penalties, dist_to_tcp = [], [], []
        for o in out:
            positions.append(o[0])
            penalties.append(o[1])
            dist_to_tcp.append(o[2])
        return np.array(positions), np.array(penalties), np.array(dist_to_tcp)

    def reset_robot_pos(self):
        out = self.execute.map(worker, self.blank, self.envs, self.reset_robot_cmds)

    def pos_behind_box(self, **args):
        return list(
            self.execute.map(worker, args, self.blank, self.envs, self.pos_behind_box_cmds)
        )

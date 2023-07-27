import fancy_gym
import numpy as np
from typing import List, Sequence, Optional, Union
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed


def make_slasync_env(env_id: str, seed: int, **kwargs: dict):
    def _init():
        env = fancy_gym.make(env_id, **kwargs)
        env.reset(seed)
        return env
    set_random_seed(seed)
    return _init


class SLAsyncBoxPushingBinEnv(SubprocVecEnv):
    def robot_state(self):
        return np.array(self.env_method(method_name="robot_state"))

    def pos_behind_box(self):
        return np.array(self.env_method(method_name="pos_behind_box"))

    def get_obs(self):
        return np.array(self.env_method(method_name="get_obs"))

    def set_tcp_pos(self, positions, hard_set):
        kwargs = {"desired_tcp_pos": positions, "hard_set": [hard_set] * self.num_envs}
        return self.env_method(
            method_name="set_tcp_pos",
            **kwargs,
        )

    def img_to_world(self, pixel_positions):
        kwargs = {"pixel_pos": pixel_positions}
        return self.env_method(method_name="img_to_world", **kwargs)

    def num_boxes_in(self):
        return np.array(self.env_method(method_name="num_boxes_in"))

    def reset_robot_pos(self):
        return self.env_method(method_name="reset_robot_pos")

    def reset_mp(self):
        return self.env_method(method_name="reset_mp")

    def render(self, **kwargs):
        if kwargs["mode"] == "obs_human":
            return self.env_method(
                method_name="render",
                **kwargs,
            )

        return np.array(self.env_method(
            method_name="render",
            **kwargs,
        ))

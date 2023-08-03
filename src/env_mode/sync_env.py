import gym
import fancy_gym
import numpy as np
from typing import Any, List, Sequence, Optional, Union
from gym.error import AlreadyPendingCallError


class SyncBoxPushingBinEnv(gym.vector.SyncVectorEnv):
    def call(self, name, *args, **kwargs) -> tuple:
        """
        Overwrite the method from SyncVectorEnv in order to specify if the arguments
        should be the same for each environment or different arguments are passed in a
        list, one element for each environment. Return tuple of lists of length num envs.

        Returns:
            (tuple): a tuple of lists where each list has self.num_envs elements
        """
        results, batch_of_args = [], kwargs.pop("batch")
        args_, kwargs_ = args, kwargs.copy()
        for i, env in enumerate(self.envs):
            function = getattr(env, name)
            if batch_of_args:
                args_ = tuple([a_[i] for a_ in args])
                kwargs_ = dict([(k, v[i]) for k, v in kwargs.items()])
            if callable(function):
                results.append(function(*args_, **kwargs_))
            else:
                results.append(function)

        if isinstance(results[0], tuple):
            results_ = [[] for _ in range(len(results[0]))]
            for r in results:
                for j, val in enumerate(r):
                    results_[j].append(val)
            results = results_

        return tuple(results)

    def robot_state(self):
        return np.array(self.call(name="robot_state", batch=False))

    def pos_behind_box(self, pos=None, total_pos=1):
        return np.array(self.call(
            name="pos_behind_box",
            batch=True,
            pos=pos,
            total_pos=total_pos,
        ))

    def get_obs(self):
        return np.array(self.call(name="get_obs", batch=False))

    def set_tcp_pos(self, positions, hard_set):
        return self.call(
            name="set_tcp_pos",
            batch=True,
            desired_tcp_pos=positions,
            hard_set=[hard_set] * self.num_envs
        )

    def img_to_world(self, pixel_positions):
        return self.call(name="img_to_world", batch=True, pixel_pos=pixel_positions)

    def num_boxes_in(self):
        return np.array(self.call(name="num_boxes_in", batch=False))

    def reset_robot_pos(self):
        return self.call(name="reset_robot_pos", batch=False)

    def reset_mp(self):
        return self.call(name="reset_mp", batch=False)

    def render(self, **kwargs):
        return np.array(self.call(
            name="render",
            batch=False,
            **kwargs,
        ))

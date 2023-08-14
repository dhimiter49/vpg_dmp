from typing import Optional, Union
import gym
import numpy as np
from gym.error import AlreadyPendingCallError


class AsyncBoxPushingBinEnv(gym.vector.AsyncVectorEnv):
    def call_async(self, name: str, *args, **kwargs):
        """
        Overwrite the method from AsyncVectorEnv in order to specify if the arguments
        should be the same for each environment or different arguments are passed in a
        list, one element for each environment.
        """
        self._assert_is_running()
        if self._state != gym.vector.async_vector_env.AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `call_async` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )
        batch_of_args = kwargs.pop("batch")
        args_, kwargs_ = args, kwargs.copy()
        for i, pipe in enumerate(self.parent_pipes):
            if batch_of_args:
                args_ = tuple([a_[i] for a_ in args])
                kwargs_ = dict([(k, v[i]) for k, v in kwargs.items()])
            pipe.send(("_call", (name, args_, kwargs_)))
        self._state = gym.vector.async_vector_env.AsyncState.WAITING_CALL

    def call_wait(self, timeout: Optional[Union[int, float]] = None) -> list:
        """
        Overwrite the method from AsyncVectorEnv in order to restructure the output.

        Return:
            (tuple): a tuple of lists where each list has self.num_envs elements
        """
        results = super().call_wait(timeout)
        if isinstance(results[0], tuple):
            results_ = [[] for _ in range(len(results[0]))]
            for r in results:
                for j, val in enumerate(r):
                    results_[j].append(val)
            results = tuple(results_)

        return results

    def robot_state(self):
        return np.array(self.call(name="robot_state", batch=False))

    def pos_behind_box(self, pos=None, total_pos=1):
        return np.array(self.call(
            name="pos_behind_box",
            batch=pos is not None,
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

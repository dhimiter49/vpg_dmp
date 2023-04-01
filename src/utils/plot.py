import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


def show_image(img_tensor):
    """
    Show image tensors. Includes error handling for shape and values.

    Args:
        img_tensor (torch.tensor): image tensor that includes at least (c, w, h)
    """
    assert img_tensor.dim() >= 3
    if img_tensor.shape[-3] != 3:  # check rgb dimension
        img_tensor = (
            img_tensor.permute(2, 0, 1)
            if img_tensor.dim() == 3
            else img_tensor.permute(0, 3, 1, 2)
        )
    img_tensor = img_tensor / 255 if img_tensor.max().item() > 1 else img_tensor
    img_grid = torchvision.utils.make_grid(img_tensor)
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.show()


def plot_step_obs(env, obs_size, pos, world_pos):
    print("Initial position: ", pos)
    depth = env.render(
        mode="depth_array", width=obs_size, height=obs_size, camera_name="rgbd"
    ).copy()
    env.data.site("ref").xpos = world_pos  # Draw pixel in the world
    rgb = env.render(
        mode="rgb_array", width=obs_size, height=obs_size, camera_name="rgbd"
    ).copy()

    rgb[tuple(pos[:2].astype(int))] = torch.ones(3)  # Draw pixel
    depth[tuple(pos[:2].astype(int))] = 0.98  # Draw depth pixel

    fig = plt.figure(figsize=(12, 6))
    fig.add_subplot(1, 2, 1)
    plt.imshow(rgb)
    fig.add_subplot(1, 2, 2)
    plt.imshow(depth)
    plt.show()

    # Hider reference point again
    env.data.site("ref").xpos = np.array([0.5, 0.0, -0.2])

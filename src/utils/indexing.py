import torch
import numpy as np
from torch import nn


def unravel_index(index, tensor):
    """
    Does the same as the numpy unravel_index function for tensors, calculate the indexes
    of a tensor from the given flattened index.

    Args:
        index (int): index of the element for flattened tensor
        tensor (torch.tensor): tensor from where to retrieve index
    Return:
        (tuple): tuple of the tensor index
    """
    idxs = ()
    for dim in reversed(tensor.shape):
        idxs += (index % dim,)
        index = index // dim

    return tuple(reversed(idxs))


def move_from_border(position, min_dist_border, limit):
    """
    Move the passed postion if it is too near the border.
    """
    x = min(max(min_dist_border, position[0]), limit - min_dist_border - 1)
    y = min(max(min_dist_border, position[1]), limit - min_dist_border - 1)
    return (x, y)


def crop_upsample(rgb, pos, obs_size):
    cropped_rgb = torch.zeros(rgb.shape)
    for i, (r, p) in enumerate(zip(rgb, pos)):
        cropped_rgb[i] = crop_upsample_single(r, p, obs_size)
    return cropped_rgb


def crop_upsample_single(rgb, pos, obs_size):
    """
    Crop an rgb input image around the position given with the given crop size.

    Args:
        rgb (np.array): array of rgb image with dim (3, width, height)
        pos (tuple): xy coordinates in the image
        obs_size (int): width=height of the obs image
    Return:
        (np.array): cropped image
    """
    crop_size = obs_size // 2.5
    img_shape = rgb.shape[:2]
    assert img_shape[0] == img_shape[1] > crop_size
    assert img_shape[0] > pos[0] and 0 <= pos[0]
    assert img_shape[1] > pos[1] and 0 <= pos[1]

    left = np.clip(pos[1] - crop_size // 1.11, 0, img_shape[1])
    right = np.clip(pos[1] + crop_size - crop_size // 1.11, 0, img_shape[1])
    up = np.clip(pos[0] - crop_size // 2, 0, img_shape[0])
    down = np.clip(pos[0] + crop_size - crop_size // 2, 0, img_shape[0])
    if down - up != crop_size:
        if up == 0:
            down = crop_size
        if down == img_shape[0]:
            up = img_shape[0] - crop_size
    if right - left != crop_size:
        if left == 0:
            right = crop_size
        if right == img_shape[1]:
            left = img_shape[1] - crop_size

    rgb = nn.Upsample(size=obs_size)(
        rgb[int(up):int(down), int(left):int(right), :].permute(2, 0, 1).unsqueeze(0)
    )
    return rgb[0].permute(1, 2, 0)


def get_neigh(position, neigh_size, limit):
    """
    Get range in x and y axis for neighbourhood aorund position.

    Args:
        position (tuple): position in 2d space
        neigh_size (int): size of the neighbourhood
        limit (int): limit or size of the 2d image
    """
    left = min(max(position[1] - neigh_size // 2, 0), limit)
    right = min(max(position[1] + 1 + neigh_size // 2, 0), limit)
    up = min(max(position[0] - neigh_size // 2, 0), limit)
    down = min(max(position[0] + 1 + neigh_size // 2, 0), limit)
    return int(up), int(down), int(left), int(right)

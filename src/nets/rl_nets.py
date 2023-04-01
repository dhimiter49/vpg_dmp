import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent

import nets.utils as utils
from .backbone import (
    UNet, DenseNetUNet, DenseNetUpsample, DenseNetFeatures, DoubleCriticWrapper
)


class VPGCritic(nn.Module):
    def __init__(
        self,
        n_orientations: int = 16,
        obs_size: int = 144,
        push_only: bool = False,
        grasp_only: bool = False,
        no_depth: bool = False,
        backbone_params: dict = {},
        device: torch.device = torch.device("cpu"),
    ):
        """
        The critic for direct pixel level evaluation based on the implementation for VPG
        (https://github.com/andyzeng/visual-pushing-grasping/blob/master/models.py). The
        architecture is made up of DenseNets for feature extraction of rgb and depth
        input and two heads for evaluating push and grasp tasks.

        Args:
            n_orientations (int): number of orientations
            obs_size (int): resolution used for the observation, including padding
            push_only (bool): only push task is optimized (evaluated and improved)
            grasp_only (bool): only grasp task is optimized (evaluated and improved)
            device (torch.device): torch device to use
        """
        super().__init__()
        assert (not grasp_only or not push_only) == True,\
            "Either only push or only grasp is possible!"
        self.push_only = push_only
        self.grasp_only = grasp_only
        self.no_depth = no_depth
        self.device = device
        self.n_orient = n_orientations
        self.rot_thetas = [i * 2 * np.pi / self.n_orient for i in range(self.n_orient)]
        self.n_primitives = (not self.push_only) + (not self.grasp_only)

        model_name = backbone_params.pop("model")
        smooth_size = backbone_params.pop("smooth_size")
        if model_name == "unet":
            model, backbone_params["n_channels"] = UNet, 3
        elif model_name == "unet_densenet_enc":
            model = DenseNetUNet
        elif model_name == "densenet":
            model, backbone_params["input_size"] = DenseNetUpsample, obs_size
        elif model_name == "double":
            model = DoubleCriticWrapper

        if not self.grasp_only:
            self.push_critic = model(**backbone_params).to(device=self.device)
        if not self.push_only:
            self.grasp_critic = model(**backbone_params).to(device=self.device)
        self.smooth = utils.create_gauss_filter(smooth_size, self.device)

        self.rot_mat_gen = lambda theta : torch.from_numpy(np.asarray([
            [[np.cos(theta)], [np.sin(theta)], [0]],
            [[-np.sin(theta)], [np.cos(theta)], [0]]
        ])).permute(2, 0, 1)  # shape (1, 2, 3)

    def forward(self, rgb, depth, rot_thetas=None, batch_thetas=False):
        """
        Critic forward pass.

        Args:
            rgb (torch.Tensor): image tensor of size (batch, c, w, h)
            depth (torch.Tensor): depth tensor of size (batch, c, w, h), all c are equal
            rot_theta (list): index of the specific theta rotation to use
            batch_thetas (bool): if the passed thetas are different for each batch, in
                this case the thetas should be applied corresponding to the batch
        Return:
            (torch.Tensor): expected reward for each rgb for all tasks in all possible
                orientations, which means size (batch, w, h, 2 * self.n_orient).
        """
        batch = rgb.shape[0]
        assert not batch_thetas or len(rot_thetas) == batch

        n_orient = self.n_orient if rot_thetas is None else len(rot_thetas)
        n_orient = 1 if batch_thetas else n_orient
        rot_thetas = self.rot_thetas if rot_thetas is None else rot_thetas
        rgb = torch.cat([rgb] * n_orient, dim=0).to(self.device)
        depth = torch.cat([depth] * n_orient, dim=0).to(self.device)
        output = torch.empty(
            (self.n_primitives, n_orient * batch) + rgb.shape[-2:], dtype=torch.float
        ).to(device=self.device)

        # Compute sample grid for rotation BEFORE neural network
        affine_mat = torch.cat([self.rot_mat_gen(-t) for t in rot_thetas], dim=0)
        affine_mat = affine_mat.to(self.device, dtype=torch.float)
        if not batch_thetas:
            affine_mat = affine_mat.repeat_interleave(batch, 0)

        # Rotate images clockwise
        with torch.no_grad():
            flow_grid = F.affine_grid(affine_mat, rgb.shape)
            rot_rgb = F.grid_sample(rgb, flow_grid, mode='nearest', align_corners=True)
            rot_depth = F.grid_sample(
                depth, flow_grid, mode='nearest', align_corners=True
            )

        # Compute intermediate features
        if not self.grasp_only:
            push_feat = self.push_critic.features(rot_rgb, rot_depth)
        if not self.push_only:
            grasp_feat = self.grasp_critic.features(rot_rgb, rot_depth)

        # Compute sample grid for rotation AFTER feature extraction
        affine_mat_inv = torch.cat([self.rot_mat_gen(t) for t in rot_thetas], dim=0)
        affine_mat_inv = affine_mat_inv.to(self.device, dtype=torch.float)
        if not batch_thetas:
            affine_mat_inv = affine_mat_inv.repeat_interleave(batch, dim=0)

        with torch.no_grad():
            if self.grasp_only:
                flow_grid = F.affine_grid(affine_mat_inv, grasp_feat.data.shape)
            else:
                flow_grid = F.affine_grid(affine_mat_inv, push_feat.data.shape)

        # Forward pass, undo rotation on output predictions, upsample results
        if not self.grasp_only:
            output[0] = self.push_critic.upsample(
                F.grid_sample(
                    self.push_critic.critic(push_feat),
                    flow_grid,
                    mode='nearest',
                    align_corners=True
                )
            ).squeeze(1)  # remove channels dimension
        if not self.push_only:
            output[self.n_primitives - 1] = self.grasp_critic.upsample(
                F.grid_sample(
                    self.grasp_critic.critic(grasp_feat),
                    flow_grid,
                    mode='nearest',
                    align_corners=True
                )
            ).squeeze(1)
        return output.view((self.n_primitives, n_orient, batch,) + rgb.shape[2:])


class VPG_DMPCritic(VPGCritic):
    def __init__(
        self,
        n_orientations: int = 1,
        obs_size: int = 144,
        push_only: bool = True,
        grasp_only: bool = False,
        no_depth: bool = True,
        backbone_params: dict = {},
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            n_orientations, obs_size,
            push_only, grasp_only, no_depth,
            backbone_params,
            device
        )

    def forward(self, rgb, depth, *args):
        """
        Same as VPGCritic forward function after omitting rotation.
        """
        batch = rgb.shape[0]
        output = torch.empty(
            (self.n_primitives, batch) + rgb.shape[-2:], dtype=torch.float
        ).to(device=self.device)

        if not self.grasp_only:
            output[0] = self.smooth(self.push_critic(rgb, depth)).squeeze(1)
        if not self.push_only:
            output[self.n_primitives - 1] =\
                self.smooth(self.grasp_critic(rgb, depth)).squeeze(1)

        return output.view((self.n_primitives, 1, batch,) + rgb.shape[2:])


class DoubleVPG_DMPCritic(VPG_DMPCritic):
    def forward(self, rgb, depth, *args):
        """
        Same as VPGCritic forward function after omitting rotation.
        """
        batch = rgb.shape[0]
        output = torch.empty(
            (self.n_primitives, batch) + rgb.shape[-2:], dtype=torch.float
        ).to(device=self.device)
        output_ = torch.empty(
            (self.n_primitives, batch) + rgb.shape[-2:], dtype=torch.float
        ).to(device=self.device)
        if not self.grasp_only:
            output[0], output_[0] =\
                [self.smooth(e).squeeze(1) for e in self.push_critic(rgb, depth)]
        if not self.push_only:
            output[self.n_primitives - 1], output_[self.n_primitives - 1] =\
                [self.smooth(e).squeeze(1) for e in self.grasp_critic(rgb, depth)]

        return output.view((self.n_primitives, 1, batch,) + rgb.shape[2:]),\
               output_.view((self.n_primitives, 1, batch,) + rgb.shape[2:])


class PixelPolicy(VPG_DMPCritic):
    def __init__(
        self,
        n_orientations: int = 1,
        obs_size: int = 144,
        push_only: bool = True,
        grasp_only: bool = False,
        no_depth: bool = True,
        backbone_params: dict = {},
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            n_orientations, obs_size,
            push_only, grasp_only, no_depth,
            backbone_params,
            device
        )
        self.pred = nn.Softmax(dim=3).to(self.device)

    def forward(self, *args):
        logits = super().forward(*args)
        pred = self.pred(logits.view(logits.shape[:3] + (-1,)))
        return pred.view(logits.shape)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        mlp_config: dict,
        obs_dim: int = 224,
        weight_vec_dim: int = 64,
        output_activation: str = "relu",
        min_std: float = 1e-5,
        init_std: float = 1.,
        backbone_params: dict = {},
        env_obs: int = None,
        robot_state_config: dict = {},
        robot_state_dim: int = 7,
        device: torch.device = torch.device("cpu"),
    ):
        """
        The policy takes as input the current observation (image or image features) from
        the environment and returns a weight vector used to parameterize the MP.

        Args:
            obs_sim (int/tuple): input representation size, or original image size
            weight_vec_dim (int): output size of the weights used for MP parameterization
            device (torch.device): torch device to use
        """
        super().__init__()
        self.min_std = min_std
        self.init_std = init_std

        self.use_env_obs = env_obs is not None
        if self.use_env_obs:
            obs_dim = env_obs
        else:
            model_name = backbone_params.pop("model")
            if model_name == "densenet":
                model = DenseNetFeatures
                obs_dim = (obs_dim // (3 * 32 * 32)) * 1024
            else:
                model, backbone_params = nn.Flatten, {"start_dim": 1}
            self.backbone = model(**backbone_params).to(device=device)

        pre_gaussian_dim = mlp_config["hidden_dims"][-1]
        self.robot_state_fusion = robot_state_config.pop("fusion_method")
        if self.robot_state_fusion is not None:
            if self.robot_state_fusion == "late":
                config = robot_state_config["backbone_config"]
                if config["hidden_dims"] is not None:
                    self.robot_state_prep =\
                        utils.create_mlp(robot_state_dim, **config).to(device=device)
                    fuse_dim = config["hidden_dims"][-1] + mlp_config["hidden_dims"][-1]
                    config["hidden_dims"], pre_gaussian_dim = [64], 64
                    self.fuse_net = utils.create_mlp(fuse_dim, **config).to(device=device)

        self.contextual_std = True
        self.mlp = utils.create_mlp(obs_dim, **mlp_config).to(device=device)
        self.mean = nn.Linear(pre_gaussian_dim, weight_vec_dim).to(device=device)
        self.std = nn.Linear(pre_gaussian_dim, weight_vec_dim).to(device=device)

        utils.init_weights(self.mean, mlp_config["weight_init"])
        utils.init_weights(self.std, mlp_config["weight_init"])

        self.pre_act_shift = (torch.tensor(self.init_std - self.min_std).exp() - 1.).log()

    def forward_pre_gaussian(self, obs: torch.Tensor, robot_state: torch.Tensor=None):
        if self.use_env_obs:
            x = self.mlp(obs)
        elif self.robot_state_fusion == "late":
            vis_feat = self.mlp(self.backbone(obs))
            robot_state_feat = self.robot_state_prep(robot_state)
            x = self.fuse_net(torch.cat([vis_feat, robot_state_feat], -1))
        else:
            x = self.mlp(self.backbone(obs))
        return x

    def forward(
        self,
        obs: torch.Tensor,
        robot_state: torch.Tensor=None,
        a: torch.Tensor=None,
        test=False
    ):
        x = self.forward_pre_gaussian(obs, robot_state)
        mean = self.mean(x)
        std = self.std(x)
        std = nn.Softplus()(std + self.pre_act_shift) + self.min_std

        weight_vec = self.sample(mean, std) if not test else mean
        std = std.diag_embed().expand(x.shape[:-1] + (-1, -1))
        if a is not None:
            logp_weight_vec = self.log_probability((mean, std), a)
        else:
            logp_weight_vec = self.log_probability((mean, std), weight_vec)

        return weight_vec, logp_weight_vec, mean, std

    def sample(self, means, std):
        eps = torch.randn(std.shape, dtype=std.dtype, device=std.device)
        samples = means + eps * std
        return samples

    def entropy(self, gauss_params):
        _, std = gauss_params
        logdet = self.log_determinant(std)
        k = std.shape[-1]
        return .5 * (k * np.log(2 * np.e * np.pi) + logdet)

    def maha(self, mean: torch.Tensor, mean_other: torch.Tensor, std: torch.Tensor):
        std = std.diagonal(dim1=-2, dim2=-1)
        diff = mean - mean_other
        return (diff / std).pow(2).sum(-1)

    def log_determinant(self, std: torch.Tensor):
        """
        Returns the log determinant of a diagonal matrix
        Args:
            std: a diagonal matrix
        Returns:
            The log determinant of std, aka log sum the diagonal
        """
        std = std.diagonal(dim1=-2, dim2=-1)
        return 2 * std.log().sum(-1)

    def log_probability(self, p: (torch.Tensor, torch.Tensor), x: torch.Tensor):
        mean, std = p
        k = x.shape[-1]

        maha_part = self.maha(x, mean, std)
        const = np.log(2.0 * np.pi) * k
        logdet = self.log_determinant(std)

        nll = -0.5 * (maha_part + const + logdet)
        return nll

    def covariance(self, std: torch.Tensor):
        return std.pow(2)

    def is_diag(self):
        return True

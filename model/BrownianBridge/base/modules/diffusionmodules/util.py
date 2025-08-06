# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#


import os
import math
import torch
import torch.nn as nn
import numpy as np
from einops import repeat

from model.BrownianBridge.base.util import instantiate_from_config


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class HybridConditioner(nn.Module):

    def __init__(self, c_concat_config, c_crossattn_config):
        super().__init__()
        self.concat_conditioner = instantiate_from_config(c_concat_config)
        self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

    def forward(self, c_concat, c_crossattn):
        c_concat = self.concat_conditioner(c_concat)
        c_crossattn = self.crossattn_conditioner(c_crossattn)
        return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()




import torch
import torch.nn as nn

class ConditionalBatchNorm2dWithAttention(nn.Module):
    def __init__(self, num_features, condition_dim, hidden_dim=None, dropout=0.1, num_heads=8, seq_len=None, **kwargs):
        """
        Improved conditional BatchNorm module with cross-attention
        Parameters maintain interface consistency with previous versions, extra parameters can be ignored or extended later

        Args:
            num_features (int): Number of channels in input feature maps.
            condition_dim (int): Dimension of conditional input (e.g., text embeddings).
            hidden_dim (int, optional): Hidden layer dimension of MLP for generating gamma and beta, defaults to condition_dim // 2.
            dropout (float): Dropout probability in cross-attention layers.
            num_heads (int): Number of heads in multi-head attention.
            seq_len (int, optional): Optional conditional sequence length (currently not used).
            **kwargs: Additional keyword arguments (maintain backward compatibility).
        """
        super(ConditionalBatchNorm2dWithAttention, self).__init__()
        self.num_features = num_features
        self.condition_dim = condition_dim
        # Maintain interface compatibility, but currently not directly using seq_len, can be saved for future use
        self.seq_len = seq_len

        if hidden_dim is None:
            hidden_dim = condition_dim // 2

        # 1. Project image features to the dimension space of conditional input, then enter attention layer
        self.feature_proj = nn.Linear(num_features, condition_dim)
        self.proj_activation = nn.ReLU()  # Add nonlinearity, helps capture more complex feature relationships

        # 2. Cross-attention layer: use projected image features as Query, conditional input as Key and Value
        self.attention = nn.MultiheadAttention(
            embed_dim=condition_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )

        # 3. Add residual connection and LayerNorm, helps stabilize attention module training
        self.attn_norm = nn.LayerNorm(condition_dim)

        # 4. Use two-layer MLP (with nonlinear activation) to generate gamma and beta parameters, improving expressive power
        self.gamma_mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_features)
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_features)
        )

        # 5. Use standard BatchNorm2d, but remove built-in affine parameters, modulate gamma and beta with conditional information
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        # 6. Initialize weights of various linear layers, ensuring stability in early training
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.feature_proj.weight)
        if self.feature_proj.bias is not None:
            nn.init.constant_(self.feature_proj.bias, 0)
        for layer in self.gamma_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        for layer in self.beta_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x, condition):
        """
        Forward Pass

        Args:
            x (torch.Tensor): Input feature maps with shape [B, num_features, H, W].
            condition (torch.Tensor): Conditional input with shape [B, seq_len, condition_dim].

        Returns:
            torch.Tensor: Modulated feature maps.
        """
        B, C, H, W = x.size()

        # Flatten feature maps to [B, H*W, C]
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # shape: [B, H*W, num_features]

        # Project features at each spatial position to condition dimension and add nonlinear activation
        x_proj = self.proj_activation(self.feature_proj(x_flat))  # shape: [B, H*W, condition_dim]

        # Use cross-attention for conditional fusion:
        # - Query: projected image features x_proj
        # - Key, Value: conditional input condition (e.g., text token sequence)
        attn_output, _ = self.attention(query=x_proj, key=condition, value=condition)  # shape: [B, H*W, condition_dim]

        # Add residual connection, then LayerNorm (similar to Transformer design, helps gradient flow and stability)
        attn_output = self.attn_norm(attn_output + x_proj)

        # Global average pooling over all spatial positions to get global conditional information
        pooled = attn_output.mean(dim=1)  # shape: [B, condition_dim]

        # Generate gamma and beta parameters through MLP (note: gamma and beta have size [B, num_features])
        gamma = self.gamma_mlp(pooled).view(B, self.num_features, 1, 1)
        beta = self.beta_mlp(pooled).view(B, self.num_features, 1, 1)

        # Apply BatchNorm to input feature maps (no built-in affine parameters), then modulate with conditionally generated gamma and beta
        out = self.bn(x)
        out = gamma * out + beta

        return out

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np
import math
from torchvision import models

from model.utils import extract, default, extract_qkv_features
from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel, QKVAttentionLegacy, ResBlock
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.BrownianBridge.base.modules.diffusionmodules.openaimodel_cbn import CBNUNetModel






class BrownianBridgeModel(nn.Module):
    def __init__(self, model_config):
        # 构造函数
        super().__init__()
        self.model_config = model_config  # 配置文件
        # model hyperparameters
        model_params = model_config.BB.params  # 模型参数
        self.num_timesteps = model_params.num_timesteps  # 时间步数
        self.mt_type = model_params.mt_type  # 噪声类型
        self.max_var = model_params.max_var if hasattr(model_params, "max_var") else 0.5  # 最大方差
        self.eta = model_params.eta if hasattr(model_params, "eta") else 1  # 噪声参数
        self.skip_sample = model_params.skip_sample  # 是否跳过采样
        self.sample_type = model_params.sample_type  # 采样类型
        self.sample_step = model_params.sample_step  # 采样步数
        self.steps = None  # 初始化 steps 为 None
        self.register_schedule()  # 时间调度表

        # loss and objective
        self.loss_type = model_params.loss_type
        self.objective = model_params.objective
        #========
        self.gradient_loss_fn = 0

        # UNet
        self.image_size = model_params.UNetParams.image_size
        self.channels = model_params.UNetParams.in_channels
        self.condition_key = model_params.UNetParams.condition_key
        if model_params.prompt is True:
            self.denoise_fn = CBNUNetModel(**vars(model_params.UNetParams))
        else:
            self.denoise_fn = UNetModel(**vars(model_params.UNetParams))

        self.lambda_l1 = 1.0  # L1损失的权重

    # 时间步数调度表
    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)

        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError

        m_tminus = np.append(0, m_t[:-1])
        
        # # 这是原来的调度函数
        variance_t = 2. * self.max_var * (m_t - m_t ** 2) 

        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t
        

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    # 将提供的初始化函数 weight_init 应用于模型中的去噪网络 denoise_fn
    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    # 获取去噪网络 denoise_fn 的参数
    def get_parameters(self):
        return self.denoise_fn.parameters()

    # 前向传播，在给定的时间步 t 下，计算模型的损失
    def forward(self, x, y, context=None, text_embed=None):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        loss, log_dict = self.p_losses(x, y, context, t, text_embed=text_embed)
        return loss, log_dict

    def p_losses(self, x0, y, context, t, noise=None, text_embed=None):
        """
        model loss
        :param x0: encoded x_ori, E(x_ori) = x0
        :param y: encoded y_ori, E(y_ori) = y
        :param y_ori: original source domain image
        :param t: timestep
        :param noise: Standard Gaussian Noise
        :return: loss
        """
        b, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))

        x_t, objective = self.q_sample(x0, y, t, noise)
        if text_embed is not None:
            #print('we are in p_losses of BrownianBridgeModel using text_embed')
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context, text_embed=text_embed)
        else:
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)

        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError()

        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        log_dict = {
            "loss": recloss,
            "x0_recon": x0_recon
        }
        return recloss, log_dict
    

    # 采样函数，根据给定的时间步 t，生成一个噪声图像x_t
    def q_sample(self, x0, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == 'noise':
            objective = noise
        elif self.objective == 'ysubx':
            objective = y - x0
        else:
            raise NotImplementedError()

        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,
            objective
        )
    # 根据目标重构值 objective_recon，预测初始图像 x0_recon
    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        else:
            raise NotImplementedError
        return x0_recon
    
    # 采样循环，它主要用于在训练或测试过程中生成多个带噪声的中间图像。
    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs
    # 反向采样，给定x_t，y，context，时间步t，生成一个噪声图像x0_recon
    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False, text_embedding=None):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context, text_embed=text_embedding)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context, text_embed = text_embedding)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon
    # 采样循环，它主要用于在训练或测试过程中生成多个带噪声的中间
    @torch.no_grad()
    def p_sample_loop(self, y, context=None, text_embed = None, clip_denoised=True, sample_mid_step=False):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, _ = self.p_sample(x_t=img, y=y, context=context, text_embedding=text_embed ,i=i, clip_denoised=clip_denoised)
            return img
    # 采样函数
    @torch.no_grad()
    def sample(self, y, context=None,  text_embed = None ,clip_denoised=True,sample_mid_step=False):
        return self.p_sample_loop(y, context, text_embed, clip_denoised, sample_mid_step)
    
    # # 采样函数
    # @torch.no_grad()
    # def image_cross_sample(self, y, context=None, clip_denoised=True, sample_mid_step=False, app_image=None):
    #     # print('=======================')
    #     # print('We are in image_cross_sample of BrownianBridgeModel')
    #     if app_image is not None:
    #         return self.img_cross_p_sample_loop(y, context, clip_denoised, sample_mid_step, app_image)
    #     else:
    #         return self.p_sample_loop(y, context, clip_denoised, sample_mid_step)
        
    # @torch.no_grad()
    # def img_cross_p_sample_loop(self, y, context=None, clip_denoised=True, sample_mid_step=False, app_image=None):
    #     if self.condition_key == "nocond":
    #         context = None
    #     else:
    #         context = y if context is None else context

    #     if sample_mid_step:
    #         imgs, one_step_imgs = [y], []
    #         for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
    #             img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
    #             imgs.append(img)
    #             one_step_imgs.append(x0_recon)
    #         return imgs, one_step_imgs
    #     else:
    #         img = y
    #         for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
    #             if app_image is not None:
    #                 img, _ = self.img_cross_p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised, app_image=app_image)
    #             else:
    #                 img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
    #         return img

    # @torch.no_grad()
    # def img_cross_p_sample(self, x_t, y, context, i, clip_denoised=False, app_image=None):
    #     b, *_, device = *x_t.shape, x_t.device        
    #     # 检查当前时间步是否为 0，即扩散过程的最后一步。
    #     if self.steps[i] == 0: 
    #         t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            
    #         # 对 x_t 进行去噪
    #         objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
    #         x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
    #         if clip_denoised:
    #             x0_recon.clamp_(-1., 1.)
    #         return x0_recon, x0_recon
    #     else:
    #         t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
    #         n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)
            
    #         # 如果当前时间步符合条件，执行交叉注意力的计算
    #         if self.steps[i].item() % 2 == 0 and self.steps[i].item() != 0:
    #             struct_qkv = extract_qkv_features(self.denoise_fn, x_t, t, context)
    #             apparance_qkv = extract_qkv_features(self.denoise_fn, app_image, t, context)
    #             # print('len of struct_qkv:', len(struct_qkv))

    #             cross_qkv_list = []
    #             for j in range(11):
    #                 cross_qkv = Img_Cross_QKVAttentionLegacy(32)(struct_qkv[j].unsqueeze(0), apparance_qkv[j].unsqueeze(0))
    #                 # cross_qkv = Img_Cross_QKVAttentionLegacy(32)(apparance_qkv[j].unsqueeze(0), struct_qkv[j].unsqueeze(0))
    #                 # cross_qkv = Img_Cross_QKVAttentionLegacy(32)(struct_qkv[j].unsqueeze(0), struct_qkv[j].unsqueeze(0))
    #                 cross_qkv_list.append(cross_qkv)
    #             #print('len of cross_qkv_list:', len(cross_qkv_list))

    #             current_qkv_index = 0
    #             hooks = []

    #             cross_features = []
    #             apparance_features = []

    #             # 钩子函数，用于捕获每个 ResBlock 的输出并修改 apparance 的计算
    #             def resblock_hook(module, input, output):
    #                 if isinstance(module, ResBlock):
    #                     if len(cross_features) < len(apparance_features):
    #                         # ===================  通过这里调整交叉特征的权重  ===================
    #                         #modified_output = 0.5*output + 0.5 * apparance_features[len(cross_features)]
    #                         modified_output = apparance_features[len(cross_features)]
    #                         # modified_output = output
    #                         cross_features.append(output)
    #                         return modified_output
    #                     else:
    #                         cross_features.append(output)

    #             # # 注册钩子到每个 ResBlock 层
    #             # for name, layer in self.denoise_fn.input_blocks.named_modules():
    #             #     if isinstance(layer, ResBlock):
    #             #         hook = layer.register_forward_hook(resblock_hook)
    #             #         hooks.append(hook)
    #             for name, layer in self.denoise_fn.named_modules():
    #                 if isinstance(layer, ResBlock):
    #                     hook = layer.register_forward_hook(resblock_hook)
    #                     hooks.append(hook)

    #             # 通过前向传播捕获 apparance_features
    #             with torch.no_grad():
    #                 _ = self.denoise_fn(app_image, t, context)
    #                 apparance_features = cross_features.copy()

    #             cross_features.clear()

    #             # 定义钩子函数用于替换 QKVAttentionLegacy 的输出
    #             def replace_attention_output(module, input, output):
    #                 nonlocal current_qkv_index
    #                 if current_qkv_index < 20:
    #                     cross_qkv = cross_qkv_list[current_qkv_index]
    #                     current_qkv_index += 1
    #                     return cross_qkv
    #                 else:
    #                     return output

    #             objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)

    #             # 移除所有钩子
    #             for hook in hooks:
    #                 hook.remove()


    #         else:
    #             objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)   
    #         x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            


    #         if clip_denoised:
    #             x0_recon.clamp_(-1., 1.)

    #         m_t = extract(self.m_t, t, x_t.shape)
    #         m_nt = extract(self.m_t, n_t, x_t.shape)
    #         var_t = extract(self.variance_t, t, x_t.shape)
    #         var_nt = extract(self.variance_t, n_t, x_t.shape)
    #         sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
    #         sigma_t = torch.sqrt(sigma2_t) * self.eta

    #         noise = torch.randn_like(x_t)
    #         x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
    #                         (x_t - (1. - m_t) * x0_recon - m_t * y)

    #         return x_tminus_mean + sigma_t * noise, x0_recon
    
# class Img_Cross_QKVAttentionLegacy(nn.Module):
#     """
#     A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
#     """

#     def __init__(self, n_heads):
#         super().__init__()
#         self.n_heads = n_heads

#     def forward(self, struct_qkv, app_qkv):
#         """
#         Apply QKV attention.
#         :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs. H is the number of heads., C is the number of channels.
#         :return: an [N x (H * C) x T] tensor after attention.
#         """
#         struct_qkv = struct_qkv[0]
#         app_qkv = app_qkv[0]
#         bs, width, length = struct_qkv.shape # [N x (H * 3 * C) x T] bs is batch size
#         assert width % (3 * self.n_heads) == 0 # head * 3 * channels 必须被整除
#         ch = width // (3 * self.n_heads) # channels
#         q_s, k_s, v_s = struct_qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
#         q_a, k_a, v_a = app_qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1) 
#         # split qkv into q, k, v， qkv shape: [N x (H * 3 * C) x T]-->[N x H ， 3 x C ， T]
#         scale = 1 / math.sqrt(math.sqrt(ch)) # scale，使用 ch 的平方根的平方根来缩放 q 和 k，使计算更加稳定（特别是对于 float16 类型的数据）
#         weight_s = torch.einsum(
#             "bct,bcs->bts", q_s * scale, k_s * scale
#         )  # More stable with f16 than dividing afterwards
#         weight_a = torch.einsum(
#             "bct,bcs->bts", q_a * scale, k_a * scale
#         )
#         weight_cross = torch.einsum(
#             "bct,bcs->bts", q_s * scale, k_a * scale
#         )
        
#         # weight = torch.softmax(weight_s.float(), dim=-1).type(weight_s.dtype) # softmax（q * scale * k * scale）, dim=-1 表示对最后一个维度进行 softmax
#         weight_cross = torch.softmax(weight_cross.float(), dim=-1).type(weight_cross.dtype)

#         mean_weight_cross = torch.mean(weight_cross)
#         #weight_contrast =(weight_cross- mean_weight_cross * weight_cross) * 1.67 + mean_weight_cross * weight_cross
#         # a_s = torch.einsum("bts,bcs->bct", weight, v_s) # aggregated output feature，[bs * n_heads, ch, length]
#         a_cross = torch.einsum("bts,bcs->bct", weight_cross, v_a)
        
#         return a_cross.reshape(bs, -1, length)
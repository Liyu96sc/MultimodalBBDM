from inspect import isfunction
import torch


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# 定义一个函数用于提取 UNet 的 QKV 中间特征图
def extract_qkv_features(unet, input_image, timestep, context):
    qkv_features = []
    hooks = []  # 用于保存钩子的句柄

    # 钩子函数用于捕获 QKV 的输出
    def qkv_hook(module, input, output):
        if hasattr(module, 'qkv'):
            # 打印调试信息
            #print(f"Hooked QKV layer in module: {module}")
            # 将输入的 4D 张量 [batch_size, channels, height, width] 变为 3D 张量 [batch_size, channels, height * width]
            input_reshaped = input[0].view(input[0].shape[0], input[0].shape[1], -1)
            # 计算 qkv 输出
            qkv = module.qkv(input_reshaped)
            qkv_features.append(qkv)

    # 注册钩子函数到所有包含 QKV 的注意力块
    for name, layer in unet.named_modules():
        # 检查层是否为 AttentionBlock，并包含 qkv 层
        if isinstance(layer, torch.nn.Module) and hasattr(layer, 'qkv'):
            #print(f"Registering hook on layer: {name}")
            hook_handle = layer.register_forward_hook(qkv_hook)
            hooks.append(hook_handle)  # 保存钩子的句柄

    # 前向传播捕获特征
    with torch.no_grad():
        _ = unet(input_image, timestep, context)

    # 移除所有注册的钩子
    for hook in hooks:
        hook.remove()

    return qkv_features
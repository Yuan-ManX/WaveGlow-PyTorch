import torch
from torch.autograd import Variable
import torch.nn.functional as F


# 使用 TorchScript 编译函数以加快运行速度
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """
    对两个输入张量进行逐元素相加，然后对结果应用 tanh 和 sigmoid 激活函数，
    最后将两个激活函数的结果逐元素相乘。

    参数:
        input_a (Tensor): 第一个输入张量，形状为 (B, C, T)。
        input_b (Tensor): 第二个输入张量，形状为 (B, C, T)。
        n_channels (Tensor): 通道数张量，用于确定 tanh 和 sigmoid 激活函数的通道范围。

    返回:
        Tensor: 激活函数处理后的输出张量，形状为 (B, C/2, T)。
    """
    # 获取通道数
    n_channels_int = n_channels[0]
    # 对输入张量进行逐元素相加
    in_act = input_a + input_b

    # 对前一半通道应用 tanh 激活函数
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    # 对后一半通道应用 sigmoid 激活函数
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    # 将两个激活函数的结果逐元素相乘
    acts = t_act * s_act
    # 返回激活函数处理后的输出张量
    return acts


class Invertible1x1Conv(torch.nn.Module):
    """
    Invertible1x1Conv 类实现了一个可逆的 1x1 卷积层。
    该层输出卷积结果及其权重矩阵的行列式对数。如果 reverse=True，则执行逆卷积操作。

    参数说明:
        c (int): 输入和输出的通道数。
    """

    def __init__(self, c):
        """
        初始化 Invertible1x1Conv 类实例。

        参数:
            c (int): 输入和输出的通道数。
        """
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(
            c, c, kernel_size=1, stride=1, padding=0, bias=False # 定义 1x1 卷积层
        )

        # 采样一个随机的正交矩阵来初始化权重
        W = torch.linalg.qr(torch.FloatTensor(c, c).normal_())[0]

        # 确保行列式的值为 1.0 而不是 -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        # 重塑为卷积权重形状
        W = W.view(c, c, 1)
        # 赋值给卷积层的权重
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        """
        前向传播方法，执行卷积操作或逆卷积操作。

        参数:
            z (Tensor): 输入张量，形状为 (B, C, T)。
            reverse (bool, 可选): 是否执行逆卷积操作，默认为 False。

        返回:
            Union[Tensor, Tuple[Tensor, Tensor]]: 如果 reverse=True，则返回张量；否则，返回一个元组，包含卷积结果和权重矩阵的行列式对数。
        """
        # 获取输入张量的形状
        batch_size, group_size, n_of_groups = z.size()

        # 获取卷积层的权重
        W = self.conv.weight.squeeze()

        if reverse:
            if not hasattr(self, "W_inverse"):
                # 如果没有缓存逆矩阵，则计算逆矩阵
                W_inverse = W.float().inverse()
                # 转换为变量
                W_inverse = Variable(W_inverse[..., None])
                if z.type() == "torch.cuda.HalfTensor":
                    # 如果输入是半精度，则转换为半精度
                    W_inverse = W_inverse.half()
                # 缓存逆矩阵
                self.W_inverse = W_inverse
            # 执行逆卷积操作
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            # 返回逆卷积结果
            return z
        else:
            # 计算权重矩阵的行列式对数
            log_det_W = batch_size * n_of_groups * torch.logdet(W)
            # 执行卷积操作
            z = self.conv(z)
            # 返回卷积结果和行列式对数
            return z, log_det_W


class WN(torch.nn.Module):
    """
    WN 类实现了一个类似于 WaveNet 的层，用于仿射耦合层。
    与 WaveNet 的主要区别在于卷积层不需要是因果的，并且没有膨胀大小的重置。
    膨胀因子在每一层上仅加倍。

    参数说明:
        n_in_channels (int): 输入通道数。
        n_mel_channels (int): mel 频谱的通道数。
        n_layers (int): 网络层数。
        n_channels (int): 每一层的通道数。
        kernel_size (int): 卷积核大小。
    """

    def __init__(
        self, n_in_channels, n_mel_channels, n_layers, n_channels, kernel_size
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        assert n_channels % 2 == 0
        # 网络层数
        self.n_layers = n_layers
        # 每一层的通道数
        self.n_channels = n_channels
        # 输入层列表
        self.in_layers = torch.nn.ModuleList()
        # 残差和跳跃连接层列表
        self.res_skip_layers = torch.nn.ModuleList()

        # 初始卷积层
        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        # 应用权重归一化
        start = torch.nn.utils.weight_norm(start, name="weight")
        self.start = start

        # 最后的卷积层初始化为0，以帮助训练稳定性
        end = torch.nn.Conv1d(n_channels, 2 * n_in_channels, 1)
        # 权重初始化为0
        end.weight.data.zero_()
        # 偏置初始化为0
        end.bias.data.zero_()
        self.end = end

        # 条件输入卷积层
        cond_layer = torch.nn.Conv1d(n_mel_channels, 2 * n_channels * n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            # 计算当前层的膨胀因子
            dilation = 2**i
            # 计算填充大小
            padding = int((kernel_size * dilation - dilation) / 2)
            # 创建输入卷积层
            in_layer = torch.nn.Conv1d(
                n_channels,
                2 * n_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            # 应用权重归一化
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            # 添加到输入层列表中
            self.in_layers.append(in_layer)

            # 残差和跳跃连接层
            if i < n_layers - 1:
                # 如果不是最后一层，通道数为 2 倍通道数
                res_skip_channels = 2 * n_channels
            else:
                # 如果是最后一层，通道数为通道数
                res_skip_channels = n_channels

            # 创建残差和跳跃连接卷积层
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            # 应用权重归一化
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            # 添加到残差和跳跃连接层列表中
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input):
        """
        前向传播方法，执行仿射耦合层的前向计算。

        参数:
            forward_input (Tuple[Tensor, Tensor]): 输入元组，包含音频信号和梅尔频谱，形状分别为 (B, C_in, T) 和 (B, C_mel, T)。

        返回:
            Tensor: 输出张量，形状为 (B, 2 * C_in, T)。
        """
        # 解包输入元组
        audio, spect = forward_input
        # 通过初始卷积层
        audio = self.start(audio)
        # 初始化输出张量
        output = torch.zeros_like(audio)
        # 创建通道数张量
        n_channels_tensor = torch.IntTensor([self.n_channels])

        # 通过条件输入卷积层
        spect = self.cond_layer(spect)

        for i in range(self.n_layers):
            # 计算条件输入的偏移量
            spect_offset = i * 2 * self.n_channels
            # 应用激活函数
            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio),
                spect[:, spect_offset : spect_offset + 2 * self.n_channels, :],
                n_channels_tensor,
            )

            # 通过残差和跳跃连接层
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                # 更新音频信号
                audio = audio + res_skip_acts[:, : self.n_channels, :]
                # 更新输出
                output = output + res_skip_acts[:, self.n_channels :, :]
            else:
                # 更新输出
                output = output + res_skip_acts

        # 通过最后的卷积层并返回结果
        return self.end(output)


class WaveGlow(torch.nn.Module):
    """
    WaveGlow 类实现了一个基于流的生成模型，用于音频生成任务。
    该模型通过多个可逆的1x1卷积层和仿射耦合层，逐步将音频信号转换为高斯分布。
    WaveGlow 结合了高效的计算资源利用和高质量的音频生成能力，广泛应用于语音合成和音频生成任务。

    参数说明:
        cfg: 配置参数对象，包含以下字段:
            VOCODER:
                INPUT_DIM (int): mel 频谱的维度数。
                N_GROUP (int): 分组大小，必须是偶数。
                N_FLOWS (int): 流的数量。
                N_EARLY_EVERY (int): 每隔多少个流进行一次早期输出。
                N_EARLY_SIZE (int): 每次早期输出的样本数。
                N_LAYERS (int): 仿射耦合层中 WN 模块的层数。
                N_CHANNELS (int): WN 模块中的通道数。
                KERNEL_SIZE (int): WN 模块中的卷积核大小。
    """
    def __init__(self, cfg):
        super(WaveGlow, self).__init__()
        # 配置参数
        self.cfg = cfg

        # 使用转置卷积对 mel 频谱进行上采样
        self.upsample = torch.nn.ConvTranspose1d(
            self.cfg.VOCODER.INPUT_DIM, # 输入通道数
            self.cfg.VOCODER.INPUT_DIM, # 输出通道数
            1024, # 卷积核大小
            stride=256, # 步长
        )
        assert self.cfg.VOCODER.N_GROUP % 2 == 0
        # 流的数量
        self.n_flows = self.cfg.VOCODER.N_FLOWS
        # 分组大小
        self.n_group = self.cfg.VOCODER.N_GROUP
        # 每隔多少个流进行一次早期输出
        self.n_early_every = self.cfg.VOCODER.N_EARLY_EVERY
        # 每次早期输出的样本数
        self.n_early_size = self.cfg.VOCODER.N_EARLY_SIZE

        # 仿射耦合层的 WN 模块列表
        self.WN = torch.nn.ModuleList()
        # 可逆1x1卷积层列表
        self.convinv = torch.nn.ModuleList()

        # 分组的一半大小
        n_half = int(self.cfg.VOCODER.N_GROUP / 2)

        # 根据已经输出的维度大小设置每一层的尺寸
        n_remaining_channels = self.cfg.VOCODER.N_GROUP
        for k in range(self.cfg.VOCODER.N_FLOWS):
            if k % self.n_early_every == 0 and k > 0:
                # 调整一半大小
                n_half = n_half - int(self.n_early_size / 2)
                # 调整剩余通道数
                n_remaining_channels = n_remaining_channels - self.n_early_size
            # 添加可逆1x1卷积层
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            # 添加仿射耦合层的 WN 模块
            self.WN.append(
                WN(
                    n_half, # 一半大小
                    self.cfg.VOCODER.INPUT_DIM * self.cfg.VOCODER.N_GROUP, # 输入维度
                    self.cfg.VOCODER.N_LAYERS, # 层数
                    self.cfg.VOCODER.N_CHANNELS, # 通道数
                    self.cfg.VOCODER.KERNEL_SIZE, # 卷积核大小
                )
            )
        self.n_remaining_channels = n_remaining_channels  # 在推理期间有用的剩余通道数

    def forward(self, forward_input):
        """
        前向传播方法，执行 WaveGlow 的前向计算。

        参数:
            forward_input (Tuple[Tensor, Tensor]): 输入元组，包含 mel 频谱和音频信号，形状分别为 (B, n_mel_channels, frames) 和 (B, time)。

        返回:
            Tuple[Tensor, List[Tensor], List[Tensor]]: 输出元组，包含生成的音频信号、对数尺度列表和对数行列式列表。
        """
        # 解包输入元组
        spect, audio = forward_input

        # 对 mel 频谱进行上采样，使其与音频信号大小一致
        spect = self.upsample(spect)
        assert spect.size(2) >= audio.size(1)
        if spect.size(2) > audio.size(1):
            # 裁剪频谱以匹配音频长度
            spect = spect[:, :, : audio.size(1)]

        # 调整频谱张量的形状以匹配音频
        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = (
            spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)
        )

        # 调整音频张量的形状
        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        # 初始化输出音频列表
        output_audio = []
        # 初始化对数尺度列表
        log_s_list = []
        # 初始化对数行列式列表
        log_det_W_list = []

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                # 添加早期输出
                output_audio.append(audio[:, : self.n_early_size, :])
                # 调整音频以排除早期输出
                audio = audio[:, self.n_early_size :, :]
            # 通过可逆1x1卷积层
            audio, log_det_W = self.convinv[k](audio)
            # 添加对数行列式
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1) / 2)
            # 前一半音频
            audio_0 = audio[:, :n_half, :]
            # 后一半音频
            audio_1 = audio[:, n_half:, :]

            # 通过仿射耦合层的 WN 模块
            output = self.WN[k]((audio_0, spect))
            # 对数尺度
            log_s = output[:, n_half:, :]
            # 偏置
            b = output[:, :n_half, :]
            # 应用仿射变换
            audio_1 = torch.exp(log_s) * audio_1 + b
            # 添加对数尺度
            log_s_list.append(log_s)

            # 合并音频
            audio = torch.cat([audio_0, audio_1], 1)

        # 添加最终音频输出
        output_audio.append(audio)
        # 返回生成的音频、对数尺度和对数行列式
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list

    @staticmethod
    def remove_weightnorm(model):
        """
        静态方法，用于移除 WaveGlow 模型中的权重归一化。

        参数:
            model (WaveGlow): 要移除权重归一化的 WaveGlow 模型。

        返回:
            WaveGlow: 移除权重归一化后的 WaveGlow 模型。
        """
        # 获取 WaveGlow 模型实例
        waveglow = model
        for WN in waveglow.WN:
            # 移除 WN 模块中 start 卷积层的权重归一化
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)
            # 移除 WN 模块中 in_layers 列表中的所有卷积层的权重归一化
            WN.in_layers = remove(WN.in_layers)
            # 移除 WN 模块中 cond_layer 卷积层的权重归一化
            WN.cond_layer = torch.nn.utils.remove_weight_norm(WN.cond_layer)
            # 移除 WN 模块中 res_skip_layers 列表中的所有卷积层的权重归一化
            WN.res_skip_layers = remove(WN.res_skip_layers)
        # 返回移除权重归一化后的 WaveGlow 模型
        return waveglow


def remove(conv_list):
    """
    辅助函数，用于移除卷积层列表中的权重归一化。

    参数:
        conv_list (List[nn.Module]): 要移除权重归一化的卷积层列表。

    返回:
        List[nn.Module]: 移除权重归一化后的卷积层列表。
    """
    # 创建一个新的 ModuleList 用于存储移除权重归一化后的卷积层
    new_conv_list = torch.nn.ModuleList()
    
    for old_conv in conv_list:
        # 移除单个卷积层的权重归一化
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        # 将移除权重归一化后的卷积层添加到新的列表中
        new_conv_list.append(old_conv)
        
    # 返回移除权重归一化后的卷积层列表
    return new_conv_list

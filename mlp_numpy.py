from pathlib import Path  # 导入 Path，用来处理项目根目录、数据路径、CSV 路径和 JSON 路径。
from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union  # 导入类型标注工具，方便说明函数和类的输入输出类型。
import json  # 导入 json，用来读取 data_utils.py 之前保存的 normalization.json 文件。
import numpy as np  # 导入 NumPy，本次实验允许使用 NumPy 进行矩阵运算。

from data_utils import IMAGE_SIZE  # 从已有 data_utils.py 中导入统一图片尺寸，例如 (64, 64)。
from data_utils import image_to_mlp_input  # 从已有 data_utils.py 中导入图片展平函数，把 H×W×3 展平成 MLP 输入向量。
from data_utils import load_image  # 从已有 data_utils.py 中导入单张图片读取函数，负责读 JPG、转 RGB、resize、除以 255。
from data_utils import read_split_csv  # 从已有 data_utils.py 中导入 CSV 划分读取函数，用来读取 train.csv / val.csv / test.csv。

Sample = Tuple[str, int, str]  # 定义样本类型别名；每个样本由“图片路径、整数标签、类别名称”组成。


def load_normalization_params(json_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:  # 定义函数：读取训练集 RGB 均值和标准差。
    json_path = Path(json_path)  # 把输入的 JSON 路径转换成 Path 对象，方便后续检查文件是否存在。
    if not json_path.exists():  # 检查 normalization.json 是否存在。
        raise FileNotFoundError(f"找不到归一化参数文件：{json_path}")  # 如果文件不存在，就主动报错，避免后续标准化失败。
    with json_path.open("r", encoding="utf-8") as file:  # 以只读模式打开 JSON 文件，并使用 UTF-8 编码。
        params = json.load(file)  # 把 JSON 文件中的内容读取成 Python 字典。
    if "mean_rgb" not in params:  # 检查 JSON 字典中是否包含 mean_rgb 字段。
        raise KeyError("normalization.json 中缺少 mean_rgb 字段。")  # 如果缺少 mean_rgb，就说明归一化参数文件不完整。
    if "std_rgb" not in params:  # 检查 JSON 字典中是否包含 std_rgb 字段。
        raise KeyError("normalization.json 中缺少 std_rgb 字段。")  # 如果缺少 std_rgb，就说明归一化参数文件不完整。
    mean_rgb = np.asarray(params["mean_rgb"], dtype=np.float32).reshape(1, 1, 3)  # 把 RGB 均值转换成 shape 为 (1, 1, 3) 的数组，方便与图片广播相减。
    std_rgb = np.asarray(params["std_rgb"], dtype=np.float32).reshape(1, 1, 3)  # 把 RGB 标准差转换成 shape 为 (1, 1, 3) 的数组，方便与图片广播相除。
    std_rgb = np.maximum(std_rgb, 1e-7)  # 防止标准差过小导致除以 0 或数值爆炸。
    return mean_rgb, std_rgb  # 返回训练集 RGB 均值和标准差，后续训练、验证、测试都必须共用它们。


def load_sample_for_mlp(sample: Sample, mean_rgb: np.ndarray, std_rgb: np.ndarray, image_size: Tuple[int, int] = IMAGE_SIZE) -> Tuple[np.ndarray, int]:  # 定义函数：把一个样本读取成 MLP 输入向量和整数标签。
    image_path, label_id, class_name = sample  # 把样本拆成图片路径、整数标签、类别名称三个变量。
    img_array = load_image(image_path, image_size=image_size, normalize=True)  # 读取图片，得到已经除以 255 的 H×W×3 RGB 数组。
    img_array = (img_array - mean_rgb) / std_rgb  # 使用训练集 RGB 均值和标准差对图片做标准化。
    x = image_to_mlp_input(img_array)  # 把标准化后的 H×W×3 图片展平成一维 MLP 输入向量。
    y = int(label_id)  # 把标签转换成 Python 整数，方便交叉熵损失函数使用。
    return x, y  # 返回一个输入向量 x 和一个整数标签 y。


def make_batch(samples: Sequence[Sample], indices: np.ndarray, mean_rgb: np.ndarray, std_rgb: np.ndarray, image_size: Tuple[int, int] = IMAGE_SIZE) -> Tuple[np.ndarray, np.ndarray]:  # 定义函数：根据一组样本索引读取一个 mini-batch。
    if len(indices) == 0:  # 检查当前 batch 的索引数量是否为 0。
        raise ValueError("indices 为空，无法构造 mini-batch。")  # 如果索引为空，就主动报错。
    x_list: List[np.ndarray] = []  # 创建空列表，用来存放当前 batch 中每张图片的展平输入向量。
    y_list: List[int] = []  # 创建空列表，用来存放当前 batch 中每张图片的整数标签。
    for index in indices:  # 遍历当前 batch 中的每一个样本索引。
        sample = samples[int(index)]  # 根据索引从样本列表中取出一个样本。
        x, y = load_sample_for_mlp(sample, mean_rgb, std_rgb, image_size=image_size)  # 读取当前样本并得到输入向量和标签。
        x_list.append(x)  # 把当前样本的输入向量加入输入列表。
        y_list.append(y)  # 把当前样本的整数标签加入标签列表。
    X_batch = np.stack(x_list, axis=0).astype(np.float32, copy=False)  # 把多个一维输入向量堆叠成二维矩阵，shape 为 batch_size×input_dim。
    y_batch = np.asarray(y_list, dtype=np.int64)  # 把标签列表转换成一维整数数组，shape 为 batch_size。
    return X_batch, y_batch  # 返回当前 mini-batch 的输入矩阵和标签向量。


def batch_iterator(samples: Sequence[Sample], batch_size: int, mean_rgb: np.ndarray, std_rgb: np.ndarray, shuffle: bool = True, rng: Optional[np.random.Generator] = None, image_size: Tuple[int, int] = IMAGE_SIZE) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:  # 定义 mini-batch 迭代器，每次 yield 一个 X_batch 和 y_batch。
    if batch_size <= 0:  # 检查 batch_size 是否为正整数。
        raise ValueError(f"batch_size 必须为正整数，但当前是 {batch_size}")  # 如果 batch_size 不合法，就主动报错。
    if len(samples) == 0:  # 检查样本列表是否为空。
        raise ValueError("samples 为空，无法生成 mini-batch。")  # 如果没有样本，就主动报错。
    if rng is None:  # 判断调用者是否传入随机数生成器。
        rng = np.random.default_rng(42)  # 如果没有传入随机数生成器，就创建一个默认随机数生成器。
    indices = np.arange(len(samples))  # 生成从 0 到 样本数-1 的索引数组。
    if shuffle:  # 判断是否需要打乱样本顺序。
        rng.shuffle(indices)  # 如果需要打乱，就原地随机打乱索引数组。
    for start in range(0, len(samples), batch_size):  # 从 0 开始，每隔 batch_size 取一个 mini-batch 的起点。
        end = min(start + batch_size, len(samples))  # 计算当前 mini-batch 的终点，最后一个 batch 可能不足 batch_size。
        batch_indices = indices[start:end]  # 从打乱后的索引中切出当前 batch 对应的样本索引。
        X_batch, y_batch = make_batch(samples, batch_indices, mean_rgb, std_rgb, image_size=image_size)  # 根据当前 batch 索引读取图片并组成 batch。
        yield X_batch, y_batch  # 把当前 batch 返回给外部训练循环或测试代码。


def parse_hidden_dims(hidden_dim: Union[int, Tuple[int, int], List[int]]) -> Tuple[int, int]:  # 定义函数：把隐藏层大小解析成两个隐藏层维度。
    if isinstance(hidden_dim, int):  # 判断 hidden_dim 是否是单个整数。
        if hidden_dim <= 0:  # 检查单个隐藏层维度是否为正整数。
            raise ValueError(f"hidden_dim 必须为正整数，但当前是 {hidden_dim}")  # 如果隐藏层维度不合法，就主动报错。
        return hidden_dim, hidden_dim  # 如果只给一个 hidden_dim，就让两个隐藏层使用同样的神经元个数。
    if isinstance(hidden_dim, (tuple, list)):  # 判断 hidden_dim 是否是 tuple 或 list。
        if len(hidden_dim) != 2:  # 检查 tuple/list 是否正好包含两个隐藏层维度。
            raise ValueError(f"hidden_dim 如果是 tuple/list，必须包含两个数，但当前是 {hidden_dim}")  # 如果长度不是 2，就主动报错。
        hidden_dim1 = int(hidden_dim[0])  # 取出第一个隐藏层维度，并转换成整数。
        hidden_dim2 = int(hidden_dim[1])  # 取出第二个隐藏层维度，并转换成整数。
        if hidden_dim1 <= 0:  # 检查第一个隐藏层维度是否为正整数。
            raise ValueError(f"hidden_dim1 必须为正整数，但当前是 {hidden_dim1}")  # 如果第一个隐藏层维度不合法，就主动报错。
        if hidden_dim2 <= 0:  # 检查第二个隐藏层维度是否为正整数。
            raise ValueError(f"hidden_dim2 必须为正整数，但当前是 {hidden_dim2}")  # 如果第二个隐藏层维度不合法，就主动报错。
        return hidden_dim1, hidden_dim2  # 返回两个隐藏层维度。
    raise TypeError(f"hidden_dim 必须是 int、tuple 或 list，但当前类型是 {type(hidden_dim)}")  # 如果 hidden_dim 类型不合法，就主动报错。


def weight_scale_for_layer(in_dim: int, activation: str, is_output: bool = False) -> float:  # 定义函数：根据输入维度和激活函数选择权重初始化尺度。
    if in_dim <= 0:  # 检查输入维度是否合法。
        raise ValueError(f"in_dim 必须为正整数，但当前是 {in_dim}")  # 如果输入维度不合法，就主动报错。
    if is_output:  # 判断当前层是否是最后的输出层。
        return float(np.sqrt(1.0 / in_dim))  # 输出层后面不接隐藏激活函数，使用较温和的 Xavier 风格尺度。
    activation_lower = activation.lower()  # 把激活函数名称转换成小写，方便兼容 ReLU/relu 等不同写法。
    if activation_lower == "relu":  # 判断隐藏层是否使用 ReLU 激活函数。
        return float(np.sqrt(2.0 / in_dim))  # ReLU 隐藏层通常使用 He 初始化尺度 sqrt(2/in_dim)。
    if activation_lower in ("tanh", "sigmoid"):  # 判断隐藏层是否使用 tanh 或 sigmoid 激活函数。
        return float(np.sqrt(1.0 / in_dim))  # tanh/sigmoid 隐藏层使用较温和的 Xavier 风格尺度。
    raise ValueError(f"不支持的激活函数：{activation}，可选值为 relu、tanh、sigmoid。")  # 如果激活函数名称未知，就主动报错。


class Linear:  # 定义全连接线性层，实现 out = xW + b 以及对应反向传播。
    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator, weight_scale: Optional[float] = None, name: str = "linear") -> None:  # 定义线性层初始化方法。
        if in_dim <= 0:  # 检查输入维度是否合法。
            raise ValueError(f"in_dim 必须为正整数，但当前是 {in_dim}")  # 如果输入维度不合法，就主动报错。
        if out_dim <= 0:  # 检查输出维度是否合法。
            raise ValueError(f"out_dim 必须为正整数，但当前是 {out_dim}")  # 如果输出维度不合法，就主动报错。
        self.in_dim = in_dim  # 保存输入维度，例如 EuroSAT 展平后输入维度是 12288。
        self.out_dim = out_dim  # 保存输出维度，例如隐藏层维度或类别数。
        self.name = name  # 保存当前层名称，方便后续打印和调试。
        scale = float(np.sqrt(1.0 / in_dim)) if weight_scale is None else float(weight_scale)  # 设置权重初始化尺度；如果外部没有传入，就使用默认 Xavier 风格尺度。
        self.W = (rng.standard_normal((in_dim, out_dim)) * scale).astype(np.float32)  # 初始化权重矩阵 W，shape 为 in_dim×out_dim。
        self.b = np.zeros((1, out_dim), dtype=np.float32)  # 初始化偏置向量 b，shape 为 1×out_dim，前向传播时会广播到整个 batch。
        self.x_cache: Optional[np.ndarray] = None  # 初始化前向传播缓存，用来在反向传播时记住输入 x。
        self.dW = np.zeros_like(self.W, dtype=np.float32)  # 初始化权重梯度 dW，shape 与 W 完全相同。
        self.db = np.zeros_like(self.b, dtype=np.float32)  # 初始化偏置梯度 db，shape 与 b 完全相同。

    def forward(self, x: np.ndarray) -> np.ndarray:  # 定义线性层前向传播函数。
        if x.ndim != 2:  # 检查输入 x 是否是二维矩阵。
            raise ValueError(f"{self.name}.forward 期望 x 是二维矩阵，但当前 shape 是 {x.shape}")  # 如果输入不是二维，就主动报错。
        if x.shape[1] != self.in_dim:  # 检查输入特征维度是否等于本层声明的 in_dim。
            raise ValueError(f"{self.name}.forward 输入维度错误：期望 {self.in_dim}，实际 {x.shape[1]}")  # 如果维度不匹配，就主动报错。
        self.x_cache = x  # 缓存输入 x，反向传播时需要用它计算 dW 和 dx。
        out = x @ self.W + self.b  # 执行线性变换；x 是 N×in_dim，W 是 in_dim×out_dim，out 是 N×out_dim。
        return out  # 返回线性层输出。

    def backward(self, dout: np.ndarray) -> np.ndarray:  # 定义线性层反向传播函数，输入是上游传来的梯度 dout。
        if self.x_cache is None:  # 检查是否已经执行过 forward。
            raise RuntimeError(f"{self.name}.backward 必须在 forward 之后调用。")  # 如果没有 forward 缓存，就无法反向传播。
        if dout.ndim != 2:  # 检查 dout 是否是二维矩阵。
            raise ValueError(f"{self.name}.backward 期望 dout 是二维矩阵，但当前 shape 是 {dout.shape}")  # 如果 dout 不是二维，就主动报错。
        if dout.shape[1] != self.out_dim:  # 检查 dout 的特征维度是否等于本层输出维度。
            raise ValueError(f"{self.name}.backward 梯度维度错误：期望 {self.out_dim}，实际 {dout.shape[1]}")  # 如果维度不匹配，就主动报错。
        self.dW = (self.x_cache.T @ dout).astype(np.float32, copy=False)  # 计算 loss 对 W 的梯度；loss 已经按 batch 平均，所以这里不再除以 batch_size。
        self.db = np.sum(dout, axis=0, keepdims=True).astype(np.float32, copy=False)  # 计算 loss 对 b 的梯度；对 batch 维度求和即可。
        dx = (dout @ self.W.T).astype(np.float32, copy=False)  # 计算 loss 对输入 x 的梯度，继续传给前一层。
        return dx  # 返回输入梯度 dx。


class ReLU:  # 定义 ReLU 激活函数层。
    def __init__(self) -> None:  # 定义 ReLU 初始化方法。
        self.mask: Optional[np.ndarray] = None  # 初始化布尔掩码缓存，用来记住 forward 时哪些位置大于 0。

    def forward(self, x: np.ndarray) -> np.ndarray:  # 定义 ReLU 前向传播函数。
        self.mask = x > 0  # 记录 x 中大于 0 的位置，反向传播时只有这些位置允许梯度通过。
        out = np.maximum(x, 0.0).astype(np.float32, copy=False)  # 对输入逐元素执行 max(x, 0)，负数变成 0，正数保持不变。
        return out  # 返回 ReLU 输出。

    def backward(self, dout: np.ndarray) -> np.ndarray:  # 定义 ReLU 反向传播函数。
        if self.mask is None:  # 检查是否已经执行过 forward。
            raise RuntimeError("ReLU.backward 必须在 forward 之后调用。")  # 如果没有 forward 缓存，就无法反向传播。
        dx = (dout * self.mask).astype(np.float32, copy=False)  # 对大于 0 的位置保留梯度，对小于等于 0 的位置把梯度置为 0。
        return dx  # 返回输入梯度 dx。


class Tanh:  # 定义 Tanh 激活函数层。
    def __init__(self) -> None:  # 定义 Tanh 初始化方法。
        self.out_cache: Optional[np.ndarray] = None  # 初始化输出缓存，因为 tanh 的导数可以用 forward 输出计算。

    def forward(self, x: np.ndarray) -> np.ndarray:  # 定义 Tanh 前向传播函数。
        out = np.tanh(x).astype(np.float32, copy=False)  # 逐元素计算 tanh(x)，并转换成 float32。
        self.out_cache = out  # 缓存 tanh 的输出，反向传播时会用到 1 - tanh(x)^2。
        return out  # 返回 Tanh 输出。

    def backward(self, dout: np.ndarray) -> np.ndarray:  # 定义 Tanh 反向传播函数。
        if self.out_cache is None:  # 检查是否已经执行过 forward。
            raise RuntimeError("Tanh.backward 必须在 forward 之后调用。")  # 如果没有 forward 缓存，就无法反向传播。
        dx = (dout * (1.0 - self.out_cache * self.out_cache)).astype(np.float32, copy=False)  # 根据 tanh 导数公式计算输入梯度。
        return dx  # 返回输入梯度 dx。


class Sigmoid:  # 定义 Sigmoid 激活函数层。
    def __init__(self) -> None:  # 定义 Sigmoid 初始化方法。
        self.out_cache: Optional[np.ndarray] = None  # 初始化输出缓存，因为 sigmoid 的导数可以用 forward 输出计算。

    def forward(self, x: np.ndarray) -> np.ndarray:  # 定义 Sigmoid 前向传播函数。
        x_clipped = np.clip(x, -50.0, 50.0)  # 对输入做裁剪，防止 exp(-x) 数值溢出。
        out = (1.0 / (1.0 + np.exp(-x_clipped))).astype(np.float32, copy=False)  # 逐元素计算 sigmoid(x)。
        self.out_cache = out  # 缓存 sigmoid 输出，反向传播时会用到 sigmoid(x)(1-sigmoid(x))。
        return out  # 返回 Sigmoid 输出。

    def backward(self, dout: np.ndarray) -> np.ndarray:  # 定义 Sigmoid 反向传播函数。
        if self.out_cache is None:  # 检查是否已经执行过 forward。
            raise RuntimeError("Sigmoid.backward 必须在 forward 之后调用。")  # 如果没有 forward 缓存，就无法反向传播。
        dx = (dout * self.out_cache * (1.0 - self.out_cache)).astype(np.float32, copy=False)  # 根据 sigmoid 导数公式计算输入梯度。
        return dx  # 返回输入梯度 dx。


def make_activation(name: str):  # 定义激活函数工厂函数，根据字符串创建对应激活层。
    name_lower = name.lower()  # 把激活函数名称转换成小写，方便兼容 "ReLU"、"relu" 等写法。
    if name_lower == "relu":  # 判断是否选择 ReLU 激活函数。
        return ReLU()  # 返回 ReLU 激活层实例。
    if name_lower == "tanh":  # 判断是否选择 Tanh 激活函数。
        return Tanh()  # 返回 Tanh 激活层实例。
    if name_lower == "sigmoid":  # 判断是否选择 Sigmoid 激活函数。
        return Sigmoid()  # 返回 Sigmoid 激活层实例。
    raise ValueError(f"不支持的激活函数：{name}，可选值为 relu、tanh、sigmoid。")  # 如果传入未知激活函数名称，就主动报错。


class SoftmaxCrossEntropyLoss:  # 定义 Softmax + Cross Entropy 合并损失层。
    def __init__(self, eps: float = 1e-12) -> None:  # 定义损失函数初始化方法。
        self.eps = eps  # 保存一个很小的正数，防止计算 log(0)。
        self.probs: Optional[np.ndarray] = None  # 初始化 softmax 概率缓存，反向传播时会用到。
        self.y: Optional[np.ndarray] = None  # 初始化标签缓存，反向传播时会用到。

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:  # 定义损失函数前向传播，输入 logits 和整数标签 y。
        if logits.ndim != 2:  # 检查 logits 是否是二维矩阵。
            raise ValueError(f"logits 应该是二维矩阵，但当前 shape 是 {logits.shape}")  # 如果 logits 不是二维，就主动报错。
        if y.ndim != 1:  # 检查 y 是否是一维标签数组。
            raise ValueError(f"y 应该是一维数组，但当前 shape 是 {y.shape}")  # 如果 y 不是一维，就主动报错。
        if logits.shape[0] != y.shape[0]:  # 检查 logits 的样本数是否等于 y 的样本数。
            raise ValueError(f"logits 和 y 的 batch_size 不一致：{logits.shape[0]} vs {y.shape[0]}")  # 如果样本数不一致，就主动报错。
        y = y.astype(np.int64, copy=False)  # 把标签转换成 int64，保证后续可以作为 NumPy 数组索引。
        if np.any(y < 0):  # 检查标签是否存在负数。
            raise ValueError("y 中存在负数标签，这是不合法的类别编号。")  # 如果标签为负数，就主动报错。
        if np.any(y >= logits.shape[1]):  # 检查标签是否超出类别数范围。
            raise ValueError("y 中存在大于等于类别数的标签，这是不合法的类别编号。")  # 如果标签超界，就主动报错。
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)  # 对 logits 每行减去最大值，提高 softmax 数值稳定性。
        exp_scores = np.exp(shifted_logits).astype(np.float32, copy=False)  # 对平移后的 logits 做指数运算。
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # 对每一行做归一化，得到每个类别的 softmax 概率。
        batch_indices = np.arange(logits.shape[0])  # 生成 batch 内样本索引，例如 0, 1, 2, ..., N-1。
        correct_class_probs = probs[batch_indices, y]  # 取出每个样本真实类别对应的预测概率。
        loss = -np.mean(np.log(correct_class_probs + self.eps))  # 计算平均交叉熵损失。
        self.probs = probs  # 缓存 softmax 概率，反向传播时使用。
        self.y = y  # 缓存真实标签，反向传播时使用。
        return float(loss)  # 返回 Python float 类型的 loss，方便打印。

    def backward(self) -> np.ndarray:  # 定义 SoftmaxCrossEntropyLoss 的反向传播函数。
        if self.probs is None:  # 检查是否已经执行过 forward。
            raise RuntimeError("SoftmaxCrossEntropyLoss.backward 必须在 forward 之后调用。")  # 如果没有 forward 缓存，就无法反向传播。
        if self.y is None:  # 检查标签缓存是否存在。
            raise RuntimeError("SoftmaxCrossEntropyLoss.backward 缺少标签缓存。")  # 如果没有标签缓存，就无法反向传播。
        batch_size = self.probs.shape[0]  # 取出 batch_size，即当前 batch 中样本个数。
        dlogits = self.probs.copy()  # 复制 softmax 概率，准备在其基础上构造 logits 的梯度。
        dlogits[np.arange(batch_size), self.y] -= 1.0  # 对真实类别位置减 1，这是 softmax+cross entropy 的简洁梯度形式。
        dlogits = (dlogits / batch_size).astype(np.float32, copy=False)  # 因为 forward 中 loss 对 batch 求了平均，所以梯度也要除以 batch_size。
        return dlogits  # 返回 loss 对 logits 的梯度，shape 与 logits 相同。


class ThreeLayerMLP:  # 定义真正的三层 MLP：两个隐藏层加一个输出层，即 fc1、fc2、fc3 三个可训练线性层。
    def __init__(self, input_dim: int, hidden_dim: Union[int, Tuple[int, int], List[int]], output_dim: int = 10, activation: str = "relu", seed: int = 42) -> None:  # 定义 MLP 初始化方法。
        if input_dim <= 0:  # 检查输入维度是否合法。
            raise ValueError(f"input_dim 必须为正整数，但当前是 {input_dim}")  # 如果输入维度不合法，就主动报错。
        if output_dim <= 0:  # 检查输出类别数是否合法。
            raise ValueError(f"output_dim 必须为正整数，但当前是 {output_dim}")  # 如果输出维度不合法，就主动报错。
        self.input_dim = input_dim  # 保存输入维度，例如 64×64×3=12288。
        self.hidden_dim1, self.hidden_dim2 = parse_hidden_dims(hidden_dim)  # 解析隐藏层维度；如果传入一个整数，则两个隐藏层共用同一维度。
        self.output_dim = output_dim  # 保存输出类别数，EuroSAT 是 10 类。
        self.activation_name = activation  # 保存激活函数名称，方便后续记录实验配置。
        rng = np.random.default_rng(seed)  # 创建 NumPy 随机数生成器，保证模型初始化可复现。
        scale1 = weight_scale_for_layer(input_dim, activation, is_output=False)  # 为第一层根据输入维度和激活函数选择初始化尺度。
        scale2 = weight_scale_for_layer(self.hidden_dim1, activation, is_output=False)  # 为第二层根据隐藏层维度和激活函数选择初始化尺度。
        scale3 = weight_scale_for_layer(self.hidden_dim2, activation, is_output=True)  # 为第三层输出层选择较温和的初始化尺度。
        self.fc1 = Linear(input_dim, self.hidden_dim1, rng, weight_scale=scale1, name="fc1")  # 创建第一层线性层，把输入向量映射到隐藏层 1。
        self.act1 = make_activation(activation)  # 创建第一层之后的激活函数，支持 relu、tanh、sigmoid。
        self.fc2 = Linear(self.hidden_dim1, self.hidden_dim2, rng, weight_scale=scale2, name="fc2")  # 创建第二层线性层，把隐藏层 1 映射到隐藏层 2。
        self.act2 = make_activation(activation)  # 创建第二层之后的激活函数，支持 relu、tanh、sigmoid。
        self.fc3 = Linear(self.hidden_dim2, output_dim, rng, weight_scale=scale3, name="fc3")  # 创建第三层线性输出层，把隐藏层 2 映射到类别 logits。

    def forward(self, x: np.ndarray) -> np.ndarray:  # 定义三层 MLP 的前向传播函数。
        z1 = self.fc1.forward(x)  # 输入 batch 先经过第一层线性变换，得到隐藏层 1 的 pre-activation。
        a1 = self.act1.forward(z1)  # 隐藏层 1 的 pre-activation 经过激活函数，得到隐藏层 1 的 activation。
        z2 = self.fc2.forward(a1)  # 隐藏层 1 的 activation 经过第二层线性变换，得到隐藏层 2 的 pre-activation。
        a2 = self.act2.forward(z2)  # 隐藏层 2 的 pre-activation 经过激活函数，得到隐藏层 2 的 activation。
        logits = self.fc3.forward(a2)  # 隐藏层 2 的 activation 经过第三层线性变换，得到每个类别的 logits。
        return logits  # 返回 logits；这里不做 softmax，因为 softmax 已经合并到交叉熵损失函数里。

    def backward(self, dlogits: np.ndarray) -> np.ndarray:  # 定义三层 MLP 的反向传播函数，输入是 loss 对 logits 的梯度。
        da2 = self.fc3.backward(dlogits)  # 先通过第三层线性输出层反传，得到 loss 对隐藏层 2 activation 的梯度。
        dz2 = self.act2.backward(da2)  # 再通过第二个激活函数反传，得到 loss 对隐藏层 2 pre-activation 的梯度。
        da1 = self.fc2.backward(dz2)  # 再通过第二层线性层反传，得到 loss 对隐藏层 1 activation 的梯度。
        dz1 = self.act1.backward(da1)  # 再通过第一个激活函数反传，得到 loss 对隐藏层 1 pre-activation 的梯度。
        dx = self.fc1.backward(dz1)  # 最后通过第一层线性层反传，得到 loss 对输入 x 的梯度。
        return dx  # 返回 loss 对输入 batch 的梯度，主要用于调试梯度形状。

    def named_parameters_and_grads(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:  # 定义函数：返回模型参数和对应梯度，后续 SGD 会用到。
        params = {  # 创建字典，用参数名映射到“参数本身、参数梯度”。
            "fc1.W": (self.fc1.W, self.fc1.dW),  # 保存第一层权重和第一层权重梯度。
            "fc1.b": (self.fc1.b, self.fc1.db),  # 保存第一层偏置和第一层偏置梯度。
            "fc2.W": (self.fc2.W, self.fc2.dW),  # 保存第二层权重和第二层权重梯度。
            "fc2.b": (self.fc2.b, self.fc2.db),  # 保存第二层偏置和第二层偏置梯度。
            "fc3.W": (self.fc3.W, self.fc3.dW),  # 保存第三层权重和第三层权重梯度。
            "fc3.b": (self.fc3.b, self.fc3.db),  # 保存第三层偏置和第三层偏置梯度。
        }  # 结束参数字典定义。
        return params  # 返回参数与梯度字典。

    def count_parameters(self) -> int:  # 定义函数：统计模型中可训练参数总数。
        total = self.fc1.W.size + self.fc1.b.size + self.fc2.W.size + self.fc2.b.size + self.fc3.W.size + self.fc3.b.size  # 把三层权重和偏置的元素个数相加。
        return int(total)  # 返回整数形式的参数总数。


if __name__ == "__main__":  # 只有直接运行 mlp_numpy.py 时，下面的单 batch 测试代码才会执行。
    project_dir = Path(__file__).resolve().parent  # 获取当前 mlp_numpy.py 所在目录，也就是项目根目录。
    root_dir = project_dir / "data" / "EuroSAT_RGB"  # 设置 EuroSAT_RGB 数据集根目录。
    train_csv_path = project_dir / "outputs" / "splits" / "train.csv"  # 设置训练集 CSV 路径。
    normalization_json_path = project_dir / "outputs" / "normalization.json"  # 设置训练集归一化参数 JSON 路径。
    mean_rgb, std_rgb = load_normalization_params(normalization_json_path)  # 读取训练集 RGB 均值和标准差。
    train_samples = read_split_csv(train_csv_path, root_dir)  # 从 train.csv 中读取训练集样本列表。
    rng = np.random.default_rng(20260427)  # 创建随机数生成器，用于 mini-batch shuffle。
    iterator = batch_iterator(train_samples, batch_size=8, mean_rgb=mean_rgb, std_rgb=std_rgb, shuffle=True, rng=rng)  # 创建一个训练集 mini-batch 迭代器。
    X_batch, y_batch = next(iterator)  # 从迭代器中取出第一个 mini-batch，用于测试前向传播和反向传播。
    input_dim = X_batch.shape[1]  # 从真实 batch 中读取输入维度，正常情况下应为 12288。
    hidden_dim = 64  # 临时设置一个较小的隐藏层维度，两个隐藏层都会使用 64 个神经元。
    output_dim = 10  # 设置输出类别数，EuroSAT 数据集共有 10 类。
    model = ThreeLayerMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, activation="relu", seed=42)  # 创建真正的三层 MLP：fc1、fc2、fc3。
    criterion = SoftmaxCrossEntropyLoss()  # 创建 Softmax + Cross Entropy 损失函数。
    logits = model.forward(X_batch)  # 执行前向传播，得到 shape 为 batch_size×10 的 logits。
    loss = criterion.forward(logits, y_batch)  # 根据 logits 和真实标签计算平均交叉熵损失。
    dlogits = criterion.backward()  # 从损失函数反向传播，得到 loss 对 logits 的梯度。
    dx = model.backward(dlogits)  # 把 dlogits 继续传回三层 MLP，完成 fc3、act2、fc2、act1、fc1 的反向传播。
    print("X_batch shape:", X_batch.shape)  # 打印输入 batch 的形状，应该是 (8, 12288)。
    print("y_batch shape:", y_batch.shape)  # 打印标签 batch 的形状，应该是 (8,)。
    print("logits shape:", logits.shape)  # 打印 logits 的形状，应该是 (8, 10)。
    print("loss:", loss)  # 打印当前 batch 的交叉熵损失。
    print("dlogits shape:", dlogits.shape)  # 打印 dlogits 的形状，应该和 logits 一样。
    print("dx shape:", dx.shape)  # 打印 dx 的形状，应该和 X_batch 一样。
    print("fc1.W shape:", model.fc1.W.shape)  # 打印第一层权重形状，应该是 (12288, 64)。
    print("fc1.dW shape:", model.fc1.dW.shape)  # 打印第一层权重梯度形状，应该和 fc1.W 一样。
    print("fc1.b shape:", model.fc1.b.shape)  # 打印第一层偏置形状，应该是 (1, 64)。
    print("fc1.db shape:", model.fc1.db.shape)  # 打印第一层偏置梯度形状，应该和 fc1.b 一样。
    print("fc2.W shape:", model.fc2.W.shape)  # 打印第二层权重形状，应该是 (64, 64)。
    print("fc2.dW shape:", model.fc2.dW.shape)  # 打印第二层权重梯度形状，应该和 fc2.W 一样。
    print("fc2.b shape:", model.fc2.b.shape)  # 打印第二层偏置形状，应该是 (1, 64)。
    print("fc2.db shape:", model.fc2.db.shape)  # 打印第二层偏置梯度形状，应该和 fc2.b 一样。
    print("fc3.W shape:", model.fc3.W.shape)  # 打印第三层权重形状，应该是 (64, 10)。
    print("fc3.dW shape:", model.fc3.dW.shape)  # 打印第三层权重梯度形状，应该和 fc3.W 一样。
    print("fc3.b shape:", model.fc3.b.shape)  # 打印第三层偏置形状，应该是 (1, 10)。
    print("fc3.db shape:", model.fc3.db.shape)  # 打印第三层偏置梯度形状，应该和 fc3.b 一样。
    print("parameter count:", model.count_parameters())  # 打印模型总参数量，方便对模型大小有直观认识。
    assert X_batch.ndim == 2  # 检查 X_batch 是否是二维矩阵。
    assert y_batch.ndim == 1  # 检查 y_batch 是否是一维标签数组。
    assert X_batch.shape[0] == y_batch.shape[0]  # 检查输入样本数和标签样本数是否一致。
    assert X_batch.shape[1] == 64 * 64 * 3  # 检查输入维度是否等于 64×64×3。
    assert logits.shape == (X_batch.shape[0], output_dim)  # 检查 logits 形状是否是 batch_size×类别数。
    assert dlogits.shape == logits.shape  # 检查损失函数反传得到的梯度形状是否和 logits 一致。
    assert dx.shape == X_batch.shape  # 检查模型反传得到的输入梯度形状是否和 X_batch 一致。
    assert model.fc1.dW.shape == model.fc1.W.shape  # 检查第一层权重梯度形状是否正确。
    assert model.fc1.db.shape == model.fc1.b.shape  # 检查第一层偏置梯度形状是否正确。
    assert model.fc2.dW.shape == model.fc2.W.shape  # 检查第二层权重梯度形状是否正确。
    assert model.fc2.db.shape == model.fc2.b.shape  # 检查第二层偏置梯度形状是否正确。
    assert model.fc3.dW.shape == model.fc3.W.shape  # 检查第三层权重梯度形状是否正确。
    assert model.fc3.db.shape == model.fc3.b.shape  # 检查第三层偏置梯度形状是否正确。
    print("真正三层 MLP 的单个 mini-batch 前向传播和反向传播已经跑通。")  # 如果所有 assert 都通过，就打印成功信息。
from pathlib import Path  # 导入 Path，用来处理项目目录、数据目录、输出目录和模型保存路径。
from typing import Dict, Optional, Tuple, Union  # 导入类型标注工具，用来说明函数输入输出的数据类型。
import argparse  # 导入 argparse，用来从命令行读取训练超参数，例如学习率、batch_size、hidden_dim。
import csv  # 导入 csv，用来把每个 epoch 的训练日志保存成 CSV 文件。
import json  # 导入 json，用来读取 normalization.json，也用来把模型配置信息保存进权重文件。
import numpy as np  # 导入 NumPy，本次实验允许使用 NumPy 进行矩阵运算和参数更新。

from data_utils import IMAGE_SIZE  # 从 data_utils.py 中导入统一图片尺寸，例如 (64, 64)。
from data_utils import read_split_csv  # 从 data_utils.py 中导入读取 train.csv / val.csv 的函数。
from mlp_numpy import ThreeLayerMLP  # 从 mlp_numpy.py 中导入真正的三层 MLP 模型。
from mlp_numpy import SoftmaxCrossEntropyLoss  # 从 mlp_numpy.py 中导入 Softmax + Cross-Entropy 损失函数。
from mlp_numpy import batch_iterator  # 从 mlp_numpy.py 中导入自己实现的 mini-batch 数据迭代器。
from mlp_numpy import load_normalization_params  # 从 mlp_numpy.py 中导入读取训练集 RGB 均值和标准差的函数。


class SGD:  # 定义 SGD 优化器类，用来根据梯度更新模型参数。
    def __init__(self, lr: float, weight_decay: float = 0.0) -> None:  # 定义 SGD 初始化方法，接收学习率和 L2 正则化强度。
        if lr <= 0.0:  # 检查学习率是否为正数。
            raise ValueError(f"学习率 lr 必须大于 0，但当前 lr={lr}")  # 如果学习率不合法，就主动报错。
        if weight_decay < 0.0:  # 检查 weight_decay 是否为非负数。
            raise ValueError(f"weight_decay 必须大于等于 0，但当前 weight_decay={weight_decay}")  # 如果正则化强度为负，就主动报错。
        self.lr = float(lr)  # 保存当前学习率，并转换成 Python float 类型。
        self.weight_decay = float(weight_decay)  # 保存 L2 正则化强度，并转换成 Python float 类型。

    def set_lr(self, lr: float) -> None:  # 定义方法：更新优化器当前使用的学习率。
        if lr <= 0.0:  # 检查新的学习率是否为正数。
            raise ValueError(f"学习率 lr 必须大于 0，但当前 lr={lr}")  # 如果新的学习率不合法，就主动报错。
        self.lr = float(lr)  # 把优化器内部的学习率更新为新的学习率。

    def step(self, model: ThreeLayerMLP) -> None:  # 定义方法：对模型参数执行一步 SGD 更新。
        params_and_grads = model.named_parameters_and_grads()  # 从模型中读取当前所有参数以及对应梯度。
        for name, (param, grad) in params_and_grads.items():  # 遍历每一个参数名、参数数组、梯度数组。
            if grad is None:  # 检查当前参数是否缺少梯度。
                raise RuntimeError(f"参数 {name} 的梯度是 None，请确认已经先执行 backward。")  # 如果梯度不存在，就说明训练流程有问题。
            if not np.all(np.isfinite(grad)):  # 检查梯度中是否出现 NaN 或 Inf。
                raise FloatingPointError(f"参数 {name} 的梯度中出现 NaN 或 Inf，请检查学习率或反向传播。")  # 如果梯度数值异常，就主动报错。
            if name.endswith(".W"):  # 判断当前参数是否是权重矩阵 W。
                update_direction = grad + self.weight_decay * param  # 对权重矩阵加入 L2 正则化梯度，即 dW + weight_decay * W。
            else:  # 如果当前参数不是权重矩阵，通常就是偏置 b。
                update_direction = grad  # 偏置项一般不做 L2 正则化，所以更新方向就是原始梯度。
            param -= self.lr * update_direction  # 按照 SGD 公式原地更新参数：param = param - lr * update_direction。


def parse_args() -> argparse.Namespace:  # 定义函数：读取命令行参数。
    parser = argparse.ArgumentParser(description="Train a three-layer NumPy MLP on EuroSAT_RGB.")  # 创建命令行参数解析器。
    parser.add_argument("--data_dir", type=str, default="data/EuroSAT_RGB", help="EuroSAT_RGB 数据集根目录。")  # 添加数据集路径参数。
    parser.add_argument("--train_csv", type=str, default="outputs/splits/train.csv", help="训练集划分 CSV 路径。")  # 添加训练集 CSV 路径参数。
    parser.add_argument("--val_csv", type=str, default="outputs/splits/val.csv", help="验证集划分 CSV 路径。")  # 添加验证集 CSV 路径参数。
    parser.add_argument("--normalization_json", type=str, default="outputs/normalization.json", help="训练集 RGB 均值和标准差 JSON 路径。")  # 添加归一化参数路径。
    parser.add_argument("--save_dir", type=str, default="outputs/train_run", help="训练日志和 best model 的保存目录。")  # 添加输出目录参数。
    parser.add_argument("--hidden_dim", type=int, default=64, help="隐藏层大小；如果不设置 hidden_dim2，则两个隐藏层都使用该大小。")  # 添加第一个隐藏层维度参数。
    parser.add_argument("--hidden_dim2", type=int, default=None, help="第二个隐藏层大小；如果省略，则第二个隐藏层等于 hidden_dim。")  # 添加可选的第二个隐藏层维度参数。
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh", "sigmoid"], help="隐藏层激活函数。")  # 添加激活函数参数。
    parser.add_argument("--batch_size", type=int, default=64, help="mini-batch 大小。")  # 添加 batch_size 参数。
    parser.add_argument("--epochs", type=int, default=10, help="训练 epoch 数。")  # 添加 epoch 参数。
    parser.add_argument("--lr", type=float, default=0.01, help="初始学习率。")  # 添加初始学习率参数。
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 正则化强度，也称 weight decay。")  # 添加 L2 正则化强度参数。
    parser.add_argument("--decay_rate", type=float, default=0.9, help="学习率 step decay 的衰减倍率。")  # 添加学习率衰减倍率参数。
    parser.add_argument("--decay_every", type=int, default=5, help="每隔多少个 epoch 衰减一次学习率。")  # 添加学习率衰减间隔参数。
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于模型初始化和训练集 shuffle。")  # 添加随机种子参数。
    return parser.parse_args()  # 解析命令行参数并返回结果。


def resolve_path(path_text: str, project_dir: Path) -> Path:  # 定义函数：把输入的相对路径或绝对路径统一转换成绝对 Path。
    path = Path(path_text)  # 把字符串路径转换成 Path 对象。
    if path.is_absolute():  # 判断输入的路径是否已经是绝对路径。
        return path  # 如果已经是绝对路径，就直接返回。
    return project_dir / path  # 如果是相对路径，就把它解释为相对于项目根目录的路径。


def load_json_dict(json_path: Union[str, Path]) -> Dict[str, object]:  # 定义函数：读取 JSON 文件并返回字典。
    json_path = Path(json_path)  # 把输入路径转换成 Path 对象。
    if not json_path.exists():  # 检查 JSON 文件是否存在。
        raise FileNotFoundError(f"找不到 JSON 文件：{json_path}")  # 如果文件不存在，就主动报错。
    with json_path.open("r", encoding="utf-8") as file:  # 以只读模式打开 JSON 文件。
        data = json.load(file)  # 把 JSON 文件内容读成 Python 字典。
    return data  # 返回读取到的字典。


def check_training_files(data_dir: Path, train_csv: Path, val_csv: Path, normalization_json: Path) -> None:  # 定义函数：检查训练所需文件是否存在。
    if not data_dir.exists():  # 检查 EuroSAT_RGB 数据目录是否存在。
        raise FileNotFoundError(f"数据集目录不存在：{data_dir}")  # 如果数据目录不存在，就提醒检查路径。
    if not train_csv.exists():  # 检查 train.csv 是否存在。
        raise FileNotFoundError(f"训练集 CSV 不存在：{train_csv}，请先运行 data_utils.py 生成划分文件。")  # 如果 train.csv 不存在，就提醒先运行 data_utils.py。
    if not val_csv.exists():  # 检查 val.csv 是否存在。
        raise FileNotFoundError(f"验证集 CSV 不存在：{val_csv}，请先运行 data_utils.py 生成划分文件。")  # 如果 val.csv 不存在，就提醒先运行 data_utils.py。
    if not normalization_json.exists():  # 检查 normalization.json 是否存在。
        raise FileNotFoundError(f"归一化参数文件不存在：{normalization_json}，请先运行 data_utils.py。")  # 如果 normalization.json 不存在，就提醒先运行 data_utils.py。


def get_current_lr(initial_lr: float, decay_rate: float, decay_every: int, epoch_index: int) -> float:  # 定义函数：根据 step decay 规则计算当前 epoch 的学习率。
    if initial_lr <= 0.0:  # 检查初始学习率是否合法。
        raise ValueError(f"initial_lr 必须大于 0，但当前 initial_lr={initial_lr}")  # 如果初始学习率不合法，就主动报错。
    if decay_rate <= 0.0:  # 检查学习率衰减倍率是否合法。
        raise ValueError(f"decay_rate 必须大于 0，但当前 decay_rate={decay_rate}")  # 如果衰减倍率不合法，就主动报错。
    if decay_every <= 0:  # 检查衰减间隔是否合法。
        raise ValueError(f"decay_every 必须为正整数，但当前 decay_every={decay_every}")  # 如果衰减间隔不合法，就主动报错。
    decay_times = epoch_index // decay_every  # 计算到当前 epoch 为止已经衰减了多少次；epoch_index 是从 0 开始的。
    current_lr = initial_lr * (decay_rate ** decay_times)  # 按 step decay 公式计算当前学习率。
    return float(current_lr)  # 返回当前学习率。


def l2_regularization_loss(model: ThreeLayerMLP, weight_decay: float) -> float:  # 定义函数：计算 L2 正则化项的 loss 数值。
    if weight_decay <= 0.0:  # 如果正则化强度为 0 或负数，就不添加正则项。
        return 0.0  # 返回 0，表示没有 L2 正则化损失。
    l2_sum = 0.0  # 创建变量，用来累计所有权重矩阵元素平方和。
    l2_sum += float(np.sum(model.fc1.W * model.fc1.W))  # 累加第一层权重矩阵 W1 的平方和。
    l2_sum += float(np.sum(model.fc2.W * model.fc2.W))  # 累加第二层权重矩阵 W2 的平方和。
    l2_sum += float(np.sum(model.fc3.W * model.fc3.W))  # 累加第三层权重矩阵 W3 的平方和。
    reg_loss = 0.5 * weight_decay * l2_sum  # 计算 0.5 * weight_decay * ||W||^2，方便梯度正好是 weight_decay * W。
    return float(reg_loss)  # 返回 L2 正则化损失。


def predict_labels(logits: np.ndarray) -> np.ndarray:  # 定义函数：把模型输出的 logits 转换成预测类别编号。
    if logits.ndim != 2:  # 检查 logits 是否是二维矩阵。
        raise ValueError(f"logits 应该是二维矩阵，但当前 shape={logits.shape}")  # 如果 logits 不是二维，就主动报错。
    preds = np.argmax(logits, axis=1).astype(np.int64, copy=False)  # 对每个样本取分数最大的类别作为预测类别。
    return preds  # 返回预测类别数组，shape 为 batch_size。


def compute_accuracy_from_logits(logits: np.ndarray, y: np.ndarray) -> float:  # 定义函数：根据 logits 和真实标签计算 accuracy。
    preds = predict_labels(logits)  # 先把 logits 转换成预测类别。
    correct = np.sum(preds == y)  # 统计预测类别等于真实标签的样本数量。
    accuracy = correct / y.shape[0]  # 用预测正确数量除以 batch_size，得到当前 batch 准确率。
    return float(accuracy)  # 返回 Python float 类型的准确率。


def train_one_epoch(model: ThreeLayerMLP, optimizer: SGD, train_samples: Tuple, batch_size: int, mean_rgb: np.ndarray, std_rgb: np.ndarray, rng: np.random.Generator) -> Tuple[float, float, float]:  # 定义函数：训练一个 epoch。
    criterion = SoftmaxCrossEntropyLoss()  # 创建交叉熵损失函数对象。
    total_loss_sum = 0.0  # 创建变量，用来累计包含 L2 正则项的训练总损失。
    data_loss_sum = 0.0  # 创建变量，用来累计不含 L2 正则项的交叉熵数据损失。
    correct_sum = 0  # 创建变量，用来累计训练集中预测正确的样本数。
    sample_count = 0  # 创建变量，用来累计训练过程中处理过的样本总数。
    iterator = batch_iterator(train_samples, batch_size=batch_size, mean_rgb=mean_rgb, std_rgb=std_rgb, shuffle=True, rng=rng)  # 创建训练集 mini-batch 迭代器，并在每个 epoch 内打乱训练样本。
    for X_batch, y_batch in iterator:  # 逐个读取 mini-batch。
        logits = model.forward(X_batch)  # 前向传播，得到当前 batch 每个样本的 10 类 logits。
        data_loss = criterion.forward(logits, y_batch)  # 计算当前 batch 的平均交叉熵损失。
        reg_loss = l2_regularization_loss(model, optimizer.weight_decay)  # 计算当前模型参数对应的 L2 正则化损失。
        total_loss = data_loss + reg_loss  # 把交叉熵损失和 L2 正则化损失相加，得到训练目标值。
        if not np.isfinite(total_loss):  # 检查 loss 是否出现 NaN 或 Inf。
            raise FloatingPointError("训练 loss 出现 NaN 或 Inf，请尝试减小学习率。")  # 如果 loss 数值异常，就主动报错。
        dlogits = criterion.backward()  # 对 SoftmaxCrossEntropyLoss 反向传播，得到 loss 对 logits 的梯度。
        model.backward(dlogits)  # 把 dlogits 继续反传过 fc3、act2、fc2、act1、fc1，计算所有参数梯度。
        optimizer.step(model)  # 用 SGD 根据梯度更新模型参数，同时对权重矩阵加入 weight_decay。
        batch_n = X_batch.shape[0]  # 读取当前 batch 的真实样本数，最后一个 batch 可能小于 batch_size。
        preds = predict_labels(logits)  # 根据更新前的 logits 得到当前 batch 的预测类别。
        correct_sum += int(np.sum(preds == y_batch))  # 把当前 batch 中预测正确的数量加入累计值。
        sample_count += int(batch_n)  # 把当前 batch 样本数加入累计样本数。
        total_loss_sum += float(total_loss) * batch_n  # 把当前 batch 的平均总损失乘以样本数后累计。
        data_loss_sum += float(data_loss) * batch_n  # 把当前 batch 的平均交叉熵损失乘以样本数后累计。
    avg_total_loss = total_loss_sum / sample_count  # 计算整个 epoch 的平均训练总损失。
    avg_data_loss = data_loss_sum / sample_count  # 计算整个 epoch 的平均训练交叉熵损失。
    avg_accuracy = correct_sum / sample_count  # 计算整个 epoch 的训练准确率。
    return float(avg_total_loss), float(avg_data_loss), float(avg_accuracy)  # 返回平均总损失、平均数据损失和训练准确率。


def evaluate(model: ThreeLayerMLP, samples: Tuple, batch_size: int, mean_rgb: np.ndarray, std_rgb: np.ndarray) -> Tuple[float, float]:  # 定义函数：在验证集上评估模型。
    criterion = SoftmaxCrossEntropyLoss()  # 创建交叉熵损失函数对象。
    loss_sum = 0.0  # 创建变量，用来累计验证集交叉熵损失。
    correct_sum = 0  # 创建变量，用来累计验证集中预测正确的样本数。
    sample_count = 0  # 创建变量，用来累计验证集中处理过的样本总数。
    iterator = batch_iterator(samples, batch_size=batch_size, mean_rgb=mean_rgb, std_rgb=std_rgb, shuffle=False, rng=None)  # 创建验证集 mini-batch 迭代器，验证时不打乱也可以。
    for X_batch, y_batch in iterator:  # 逐个读取验证集 mini-batch。
        logits = model.forward(X_batch)  # 前向传播，得到当前验证 batch 的 logits。
        loss = criterion.forward(logits, y_batch)  # 计算当前验证 batch 的平均交叉熵损失。
        batch_n = X_batch.shape[0]  # 读取当前验证 batch 的样本数。
        preds = predict_labels(logits)  # 根据 logits 得到当前验证 batch 的预测类别。
        correct_sum += int(np.sum(preds == y_batch))  # 累加当前 batch 中预测正确的样本数。
        sample_count += int(batch_n)  # 累加当前 batch 的样本数。
        loss_sum += float(loss) * batch_n  # 把当前 batch 的平均 loss 乘以样本数后累计。
    avg_loss = loss_sum / sample_count  # 计算验证集平均交叉熵损失。
    avg_accuracy = correct_sum / sample_count  # 计算验证集准确率。
    return float(avg_loss), float(avg_accuracy)  # 返回验证集平均 loss 和验证集 accuracy。


def save_model_npz(model: ThreeLayerMLP, save_path: Union[str, Path], metadata: Dict[str, object], mean_rgb: np.ndarray, std_rgb: np.ndarray) -> None:  # 定义函数：把模型权重和必要信息保存成 npz 文件。
    save_path = Path(save_path)  # 把保存路径转换成 Path 对象。
    save_path.parent.mkdir(parents=True, exist_ok=True)  # 确保保存目录存在；如果不存在就自动创建。
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)  # 把模型配置和实验信息转换成 JSON 字符串。
    np.savez_compressed(save_path, fc1_W=model.fc1.W, fc1_b=model.fc1.b, fc2_W=model.fc2.W, fc2_b=model.fc2.b, fc3_W=model.fc3.W, fc3_b=model.fc3.b, mean_rgb=mean_rgb.reshape(-1), std_rgb=std_rgb.reshape(-1), metadata_json=np.array(metadata_json))  # 用压缩 npz 保存三层权重、偏置、归一化参数和元信息。


def write_log_header(log_csv_path: Union[str, Path]) -> None:  # 定义函数：创建训练日志 CSV 并写入表头。
    log_csv_path = Path(log_csv_path)  # 把日志路径转换成 Path 对象。
    log_csv_path.parent.mkdir(parents=True, exist_ok=True)  # 确保日志目录存在。
    with log_csv_path.open("w", newline="", encoding="utf-8") as file:  # 以写入模式打开日志 CSV 文件。
        writer = csv.writer(file)  # 创建 CSV 写入器。
        writer.writerow(["epoch", "train_loss", "train_data_loss", "val_loss", "train_acc", "val_acc", "lr"])  # 写入每一列的列名。


def append_log_row(log_csv_path: Union[str, Path], epoch: int, train_loss: float, train_data_loss: float, val_loss: float, train_acc: float, val_acc: float, lr: float) -> None:  # 定义函数：向训练日志 CSV 追加一行记录。
    log_csv_path = Path(log_csv_path)  # 把日志路径转换成 Path 对象。
    with log_csv_path.open("a", newline="", encoding="utf-8") as file:  # 以追加模式打开日志 CSV 文件。
        writer = csv.writer(file)  # 创建 CSV 写入器。
        writer.writerow([epoch, train_loss, train_data_loss, val_loss, train_acc, val_acc, lr])  # 写入当前 epoch 的训练和验证指标。


def main() -> None:  # 定义主函数：组织完整训练流程。
    args = parse_args()  # 读取命令行参数。
    if args.batch_size <= 0:  # 检查 batch_size 是否合法。
        raise ValueError(f"batch_size 必须为正整数，但当前 batch_size={args.batch_size}")  # 如果 batch_size 不合法，就主动报错。
    if args.epochs <= 0:  # 检查 epochs 是否合法。
        raise ValueError(f"epochs 必须为正整数，但当前 epochs={args.epochs}")  # 如果 epochs 不合法，就主动报错。
    np.random.seed(args.seed)  # 设置 NumPy 全局随机种子，增强实验可复现性。
    project_dir = Path(__file__).resolve().parent  # 获取当前 train.py 所在目录，也就是项目根目录。
    data_dir = resolve_path(args.data_dir, project_dir)  # 解析数据集目录路径。
    train_csv = resolve_path(args.train_csv, project_dir)  # 解析训练集 CSV 路径。
    val_csv = resolve_path(args.val_csv, project_dir)  # 解析验证集 CSV 路径。
    normalization_json = resolve_path(args.normalization_json, project_dir)  # 解析 normalization.json 路径。
    save_dir = resolve_path(args.save_dir, project_dir)  # 解析训练输出目录路径。
    check_training_files(data_dir, train_csv, val_csv, normalization_json)  # 检查训练所需的数据文件是否都存在。
    save_dir.mkdir(parents=True, exist_ok=True)  # 创建训练输出目录；如果已经存在就不会报错。
    log_csv_path = save_dir / "train_log.csv"  # 设置训练日志 CSV 的保存路径。
    best_model_path = save_dir / "best_model.npz"  # 设置验证集最优模型权重的保存路径。
    mean_rgb, std_rgb = load_normalization_params(normalization_json)  # 读取训练集 RGB 均值和标准差。
    normalization_params = load_json_dict(normalization_json)  # 读取完整 normalization.json 字典，后续保存到模型元信息中。
    train_samples = read_split_csv(train_csv, data_dir)  # 从 train.csv 读取训练集样本列表。
    val_samples = read_split_csv(val_csv, data_dir)  # 从 val.csv 读取验证集样本列表。
    if len(train_samples) == 0:  # 检查训练集是否为空。
        raise ValueError("训练集为空，无法训练模型。")  # 如果训练集为空，就主动报错。
    if len(val_samples) == 0:  # 检查验证集是否为空。
        raise ValueError("验证集为空，无法选择 best model。")  # 如果验证集为空，就主动报错。
    class_to_idx = normalization_params.get("class_to_idx", {})  # 从 normalization.json 中读取类别名称到整数标签的映射。
    output_dim = len(class_to_idx) if isinstance(class_to_idx, dict) and len(class_to_idx) > 0 else 10  # 根据类别映射推断输出类别数；如果映射缺失则默认 10 类。
    input_dim = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3)  # 根据图片尺寸计算 MLP 输入维度，即 64×64×3=12288。
    hidden_dim_config = args.hidden_dim if args.hidden_dim2 is None else (args.hidden_dim, args.hidden_dim2)  # 如果没有 hidden_dim2，就两个隐藏层共用 hidden_dim；否则使用两个不同隐藏层维度。
    model = ThreeLayerMLP(input_dim=input_dim, hidden_dim=hidden_dim_config, output_dim=output_dim, activation=args.activation, seed=args.seed)  # 创建真正三层 MLP 模型。
    optimizer = SGD(lr=args.lr, weight_decay=args.weight_decay)  # 创建 SGD 优化器，并设置初始学习率和 weight decay。
    train_rng = np.random.default_rng(args.seed)  # 创建训练集 shuffle 使用的随机数生成器。
    write_log_header(log_csv_path)  # 初始化训练日志 CSV 文件并写入表头。
    best_val_acc = -1.0  # 初始化最佳验证集准确率为 -1，确保第一个 epoch 会保存模型。
    best_epoch = 0  # 初始化最佳 epoch 编号。
    print("数据目录：", data_dir)  # 打印数据目录，方便检查路径。
    print("训练集样本数：", len(train_samples))  # 打印训练集样本数量。
    print("验证集样本数：", len(val_samples))  # 打印验证集样本数量。
    print("模型结构：", f"{input_dim} -> {model.hidden_dim1} -> {model.hidden_dim2} -> {output_dim}")  # 打印三层 MLP 的维度结构。
    print("激活函数：", args.activation)  # 打印当前使用的激活函数。
    print("参数数量：", model.count_parameters())  # 打印模型可训练参数总数。
    print("日志路径：", log_csv_path)  # 打印训练日志保存路径。
    print("best model 路径：", best_model_path)  # 打印最佳模型保存路径。
    for epoch_index in range(args.epochs):  # 按 epoch 循环训练模型。
        epoch = epoch_index + 1  # 把从 0 开始的 epoch_index 转换成从 1 开始的 epoch 编号，方便打印。
        current_lr = get_current_lr(args.lr, args.decay_rate, args.decay_every, epoch_index)  # 根据 step decay 规则计算当前 epoch 学习率。
        optimizer.set_lr(current_lr)  # 把优化器学习率更新为当前 epoch 的学习率。
        train_loss, train_data_loss, train_acc = train_one_epoch(model, optimizer, train_samples, args.batch_size, mean_rgb, std_rgb, train_rng)  # 在训练集上训练一个 epoch。
        val_loss, val_acc = evaluate(model, val_samples, args.batch_size, mean_rgb, std_rgb)  # 在验证集上评估当前模型。
        append_log_row(log_csv_path, epoch, train_loss, train_data_loss, val_loss, train_acc, val_acc, current_lr)  # 把当前 epoch 的指标写入训练日志。
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | lr={current_lr:.6g} | train_loss={train_loss:.6f} | train_acc={train_acc:.4f} | val_loss={val_loss:.6f} | val_acc={val_acc:.4f}")  # 打印当前 epoch 的训练和验证结果。
        if val_acc > best_val_acc:  # 判断当前验证集准确率是否超过历史最好结果。
            best_val_acc = val_acc  # 如果当前验证集准确率更高，就更新历史最佳验证集准确率。
            best_epoch = epoch  # 记录当前最佳模型来自哪个 epoch。
            metadata = {"input_dim": input_dim, "hidden_dim1": model.hidden_dim1, "hidden_dim2": model.hidden_dim2, "output_dim": output_dim, "activation": args.activation, "best_epoch": best_epoch, "best_val_acc": best_val_acc, "batch_size": args.batch_size, "initial_lr": args.lr, "current_lr": current_lr, "weight_decay": args.weight_decay, "decay_rate": args.decay_rate, "decay_every": args.decay_every, "seed": args.seed, "class_to_idx": normalization_params.get("class_to_idx", {}), "idx_to_class": normalization_params.get("idx_to_class", {}), "image_size": list(IMAGE_SIZE)}  # 创建模型元信息字典，方便之后测试和复现实验。
            save_model_npz(model, best_model_path, metadata, mean_rgb, std_rgb)  # 保存当前验证集准确率最高的模型权重。
            print(f"  已保存新的 best model：epoch={best_epoch}, val_acc={best_val_acc:.4f}")  # 打印保存 best model 的提示信息。
    print("训练结束。")  # 打印训练结束提示。
    print("最佳 epoch：", best_epoch)  # 打印最佳模型对应的 epoch。
    print("最佳验证集准确率：", best_val_acc)  # 打印最佳验证集准确率。
    print("最佳模型已保存到：", best_model_path)  # 打印最佳模型保存路径。
    print("训练日志已保存到：", log_csv_path)  # 打印训练日志保存路径。


if __name__ == "__main__":  # 只有直接运行 train.py 时，才执行主训练流程。
    main()  # 调用主函数，开始训练。

from pathlib import Path  # 导入 Path，用来处理项目路径、测试集 CSV 路径、模型权重路径和输出文件路径。
from typing import Dict, List, Optional, Tuple, Union  # 导入类型标注工具，方便说明函数输入输出的数据类型。
import argparse  # 导入 argparse，用来从命令行读取 evaluate.py 的参数。
import csv  # 导入 csv，用来保存 confusion matrix、各类别准确率和预测结果。
import json  # 导入 json，用来读取 best_hyperparams.json，也用来保存测试指标。
import math  # 导入 math，用来计算可视化网格的行数。
import numpy as np  # 导入 NumPy，用来做模型前向传播、混淆矩阵统计和数组处理。
import matplotlib  # 导入 matplotlib 主模块，用来设置非交互式绘图后端。
matplotlib.use("Agg")  # 使用 Agg 后端，让脚本在没有图形界面的终端里也能保存图片。
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot，用来绘制曲线、混淆矩阵、权重图和错例图。

from data_utils import IMAGE_SIZE  # 从 data_utils.py 导入统一图片尺寸，例如 (64, 64)。
from data_utils import load_image  # 从 data_utils.py 导入单张图片读取函数，用于错例可视化时显示原图。
from data_utils import read_split_csv  # 从 data_utils.py 导入读取 test.csv 的函数。
from mlp_numpy import ThreeLayerMLP  # 从 mlp_numpy.py 导入真正三层 MLP 模型类。
from mlp_numpy import SoftmaxCrossEntropyLoss  # 从 mlp_numpy.py 导入 Softmax + Cross Entropy 损失函数。
from mlp_numpy import batch_iterator  # 从 mlp_numpy.py 导入自己实现的 mini-batch 迭代器。
from mlp_numpy import load_normalization_params  # 从 mlp_numpy.py 导入读取 normalization.json 的函数，作为权重文件缺少均值方差时的备用方案。
from train import predict_labels  # 从 train.py 导入根据 logits 得到预测类别的函数。
from train import resolve_path  # 从 train.py 导入把相对路径转换成项目根目录下绝对路径的函数。
from train import load_json_dict  # 从 train.py 导入读取 JSON 文件的函数。


Sample = Tuple[str, int, str]  # 定义样本类型别名；每个样本由“图片路径、整数标签、类别名称”组成。


def parse_args() -> argparse.Namespace:  # 定义函数：解析 evaluate.py 的命令行参数。
    parser = argparse.ArgumentParser(description="Evaluate and visualize a trained NumPy three-layer MLP on EuroSAT_RGB.")  # 创建命令行参数解析器。
    parser.add_argument("--data_dir", type=str, default="data/EuroSAT_RGB", help="EuroSAT_RGB 数据集根目录。")  # 设置 EuroSAT_RGB 根目录参数。
    parser.add_argument("--test_csv", type=str, default="outputs/splits/test.csv", help="测试集 test.csv 路径。")  # 设置测试集 CSV 路径参数。
    parser.add_argument("--weights", type=str, default="outputs/hparam_search/best_overall_model.npz", help="要加载的最优模型权重 .npz 路径。")  # 设置模型权重路径参数。
    parser.add_argument("--normalization_json", type=str, default="outputs/normalization.json", help="训练集 RGB 均值和标准差 JSON 路径；仅在权重文件不含 mean/std 时备用。")  # 设置归一化参数备用路径。
    parser.add_argument("--best_hparams_json", type=str, default="outputs/hparam_search/best_hyperparams.json", help="超参数搜索得到的 best_hyperparams.json 路径，用于自动找到最佳 run 的训练日志。")  # 设置最佳超参数 JSON 路径。
    parser.add_argument("--search_dir", type=str, default="outputs/hparam_search", help="超参数搜索输出目录，用于在 JSON 路径失效时推断最佳 run 的 train_log.csv。")  # 设置超参数搜索目录。
    parser.add_argument("--train_log", type=str, default="", help="训练日志 train_log.csv 路径；如果为空，则尝试从 best_hyperparams.json 自动读取。")  # 设置训练日志路径参数。
    parser.add_argument("--output_dir", type=str, default="outputs", help="评估结果和可视化图片的总输出目录。")  # 设置输出总目录参数。
    parser.add_argument("--batch_size", type=int, default=64, help="测试评估时使用的 mini-batch 大小。")  # 设置测试 batch size。
    parser.add_argument("--num_weight_images", type=int, default=16, help="第一层权重可视化时显示多少个隐藏单元。")  # 设置可视化隐藏单元数量。
    parser.add_argument("--num_error_examples", type=int, default=12, help="错例分析图中最多显示多少个错误分类样本。")  # 设置错例数量。
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于随机抽取错例样本。")  # 设置随机种子。
    return parser.parse_args()  # 返回解析后的命令行参数。


def read_metadata_from_npz_array(metadata_array: np.ndarray) -> Dict[str, object]:  # 定义函数：从 npz 中保存的 metadata_json 数组读取元信息字典。
    metadata_text = str(metadata_array.item())  # 把 0 维 NumPy 字符串数组转换成普通 Python 字符串。
    metadata = json.loads(metadata_text)  # 把 JSON 字符串解析成 Python 字典。
    return metadata  # 返回模型元信息字典。


def load_model_from_npz(weights_path: Union[str, Path], normalization_json_path: Union[str, Path]) -> Tuple[ThreeLayerMLP, Dict[str, object], np.ndarray, np.ndarray]:  # 定义函数：从 best_model.npz 或 best_overall_model.npz 加载模型。
    weights_path = Path(weights_path)  # 把模型权重路径转换成 Path 对象。
    if not weights_path.exists():  # 检查模型权重文件是否存在。
        raise FileNotFoundError(f"找不到模型权重文件：{weights_path}")  # 如果权重文件不存在，就主动报错。
    with np.load(weights_path, allow_pickle=False) as data:  # 打开 npz 权重文件，并禁止 pickle 以保持简单安全。
        available_keys = set(data.files)  # 读取 npz 文件中保存的所有字段名称。
        fc1_W = data["fc1_W"].astype(np.float32, copy=True)  # 读取第一层权重矩阵，并转换成 float32。
        fc1_b = data["fc1_b"].astype(np.float32, copy=True)  # 读取第一层偏置向量，并转换成 float32。
        fc2_W = data["fc2_W"].astype(np.float32, copy=True)  # 读取第二层权重矩阵，并转换成 float32。
        fc2_b = data["fc2_b"].astype(np.float32, copy=True)  # 读取第二层偏置向量，并转换成 float32。
        fc3_W = data["fc3_W"].astype(np.float32, copy=True)  # 读取第三层权重矩阵，并转换成 float32。
        fc3_b = data["fc3_b"].astype(np.float32, copy=True)  # 读取第三层偏置向量，并转换成 float32。
        metadata = read_metadata_from_npz_array(data["metadata_json"]) if "metadata_json" in available_keys else {}  # 如果 npz 中保存了元信息，就读取；否则使用空字典。
        mean_rgb_from_npz = data["mean_rgb"].astype(np.float32, copy=True).reshape(1, 1, 3) if "mean_rgb" in available_keys else None  # 如果权重文件保存了 RGB 均值，就读取并 reshape。
        std_rgb_from_npz = data["std_rgb"].astype(np.float32, copy=True).reshape(1, 1, 3) if "std_rgb" in available_keys else None  # 如果权重文件保存了 RGB 标准差，就读取并 reshape。
    input_dim = int(metadata.get("input_dim", fc1_W.shape[0]))  # 从元信息读取输入维度；若缺失则从 fc1_W 的行数推断。
    hidden_dim1 = int(metadata.get("hidden_dim1", fc1_W.shape[1]))  # 从元信息读取隐藏层 1 大小；若缺失则从 fc1_W 的列数推断。
    hidden_dim2 = int(metadata.get("hidden_dim2", fc2_W.shape[1]))  # 从元信息读取隐藏层 2 大小；若缺失则从 fc2_W 的列数推断。
    output_dim = int(metadata.get("output_dim", fc3_W.shape[1]))  # 从元信息读取输出类别数；若缺失则从 fc3_W 的列数推断。
    activation = str(metadata.get("activation", "relu"))  # 从元信息读取激活函数；若缺失则默认使用 relu。
    model = ThreeLayerMLP(input_dim=input_dim, hidden_dim=(hidden_dim1, hidden_dim2), output_dim=output_dim, activation=activation, seed=0)  # 用相同结构重新创建一个三层 MLP。
    if model.fc1.W.shape != fc1_W.shape:  # 检查第一层权重形状是否和模型结构一致。
        raise ValueError(f"fc1_W shape 不匹配：模型需要 {model.fc1.W.shape}，权重文件提供 {fc1_W.shape}")  # 如果第一层形状不一致，就报错。
    if model.fc2.W.shape != fc2_W.shape:  # 检查第二层权重形状是否和模型结构一致。
        raise ValueError(f"fc2_W shape 不匹配：模型需要 {model.fc2.W.shape}，权重文件提供 {fc2_W.shape}")  # 如果第二层形状不一致，就报错。
    if model.fc3.W.shape != fc3_W.shape:  # 检查第三层权重形状是否和模型结构一致。
        raise ValueError(f"fc3_W shape 不匹配：模型需要 {model.fc3.W.shape}，权重文件提供 {fc3_W.shape}")  # 如果第三层形状不一致，就报错。
    model.fc1.W[...] = fc1_W  # 把权重文件中的第一层权重复制进模型。
    model.fc1.b[...] = fc1_b  # 把权重文件中的第一层偏置复制进模型。
    model.fc2.W[...] = fc2_W  # 把权重文件中的第二层权重复制进模型。
    model.fc2.b[...] = fc2_b  # 把权重文件中的第二层偏置复制进模型。
    model.fc3.W[...] = fc3_W  # 把权重文件中的第三层权重复制进模型。
    model.fc3.b[...] = fc3_b  # 把权重文件中的第三层偏置复制进模型。
    if mean_rgb_from_npz is not None and std_rgb_from_npz is not None:  # 判断权重文件中是否已经保存了标准化参数。
        mean_rgb = mean_rgb_from_npz  # 如果权重文件有均值，就直接使用权重文件里的均值。
        std_rgb = np.maximum(std_rgb_from_npz, 1e-7)  # 如果权重文件有标准差，就使用它并防止除以 0。
    else:  # 如果权重文件没有保存标准化参数。
        mean_rgb, std_rgb = load_normalization_params(normalization_json_path)  # 从 normalization.json 中读取训练集均值和标准差。
    metadata["weights_path"] = str(weights_path)  # 把当前权重文件路径写入元信息，方便保存测试结果。
    return model, metadata, mean_rgb, std_rgb  # 返回加载好的模型、元信息、RGB 均值和 RGB 标准差。


def get_class_names(metadata: Dict[str, object], samples: List[Sample], num_classes: int) -> List[str]:  # 定义函数：得到每个类别编号对应的类别名称。
    idx_to_class: Dict[int, str] = {}  # 创建空字典，用来保存整数标签到类别名称的映射。
    raw_idx_to_class = metadata.get("idx_to_class", {})  # 尝试从模型元信息中读取 idx_to_class。
    if isinstance(raw_idx_to_class, dict):  # 检查读取到的 idx_to_class 是否是字典。
        for key, value in raw_idx_to_class.items():  # 遍历 JSON 中保存的标签编号和类别名称。
            idx_to_class[int(key)] = str(value)  # 把 JSON 字符串键转换成整数键。
    for sample in samples:  # 遍历测试集样本，用测试集 CSV 中的类别名称补全映射。
        image_path, label_id, class_name = sample  # 拆出图片路径、整数标签和类别名称。
        idx_to_class.setdefault(int(label_id), str(class_name))  # 如果某个标签还没有类别名，就从样本中补充。
    class_names = [idx_to_class.get(index, f"class_{index}") for index in range(num_classes)]  # 按 0 到 num_classes-1 的顺序生成类别名称列表。
    return class_names  # 返回类别名称列表。


def predict_dataset(model: ThreeLayerMLP, samples: List[Sample], batch_size: int, mean_rgb: np.ndarray, std_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:  # 定义函数：在整个测试集上预测并计算 loss/accuracy。
    criterion = SoftmaxCrossEntropyLoss()  # 创建交叉熵损失函数对象。
    y_true_list: List[np.ndarray] = []  # 创建列表，用来保存每个 batch 的真实标签。
    y_pred_list: List[np.ndarray] = []  # 创建列表，用来保存每个 batch 的预测标签。
    loss_sum = 0.0  # 创建变量，用来累计测试集交叉熵损失。
    correct_sum = 0  # 创建变量，用来累计预测正确的样本数。
    sample_count = 0  # 创建变量，用来累计测试样本总数。
    iterator = batch_iterator(samples, batch_size=batch_size, mean_rgb=mean_rgb, std_rgb=std_rgb, shuffle=False, rng=None)  # 创建测试集 batch 迭代器；测试时不打乱顺序。
    for X_batch, y_batch in iterator:  # 逐个遍历测试集 batch。
        logits = model.forward(X_batch)  # 对当前 batch 执行前向传播，得到 logits。
        loss = criterion.forward(logits, y_batch)  # 计算当前 batch 的平均交叉熵 loss。
        preds = predict_labels(logits)  # 根据 logits 取最大值类别，得到预测标签。
        batch_n = X_batch.shape[0]  # 读取当前 batch 的样本数量。
        y_true_list.append(y_batch.copy())  # 把当前 batch 的真实标签保存下来。
        y_pred_list.append(preds.copy())  # 把当前 batch 的预测标签保存下来。
        loss_sum += float(loss) * batch_n  # 把当前 batch 的平均 loss 乘以样本数后累计。
        correct_sum += int(np.sum(preds == y_batch))  # 累计当前 batch 中预测正确的样本数。
        sample_count += int(batch_n)  # 累计当前 batch 的样本数。
    y_true = np.concatenate(y_true_list, axis=0)  # 把所有 batch 的真实标签拼接成一个一维数组。
    y_pred = np.concatenate(y_pred_list, axis=0)  # 把所有 batch 的预测标签拼接成一个一维数组。
    test_loss = loss_sum / sample_count  # 计算整个测试集的平均交叉熵 loss。
    test_accuracy = correct_sum / sample_count  # 计算整个测试集的 accuracy。
    return y_true, y_pred, float(test_loss), float(test_accuracy)  # 返回真实标签、预测标签、测试 loss 和测试 accuracy。


def build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:  # 定义函数：手写构造混淆矩阵。
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)  # 创建 num_classes×num_classes 的整数矩阵，行是真实类别，列是预测类别。
    for true_label, pred_label in zip(y_true, y_pred):  # 遍历每一个样本的真实标签和预测标签。
        cm[int(true_label), int(pred_label)] += 1  # 在“真实类别行、预测类别列”的位置计数加 1。
    return cm  # 返回混淆矩阵。


def compute_class_accuracy(cm: np.ndarray) -> np.ndarray:  # 定义函数：根据混淆矩阵计算每个类别的准确率。
    correct_per_class = np.diag(cm).astype(np.float64)  # 取混淆矩阵对角线，表示每个类别预测正确的数量。
    total_per_class = np.sum(cm, axis=1).astype(np.float64)  # 对每一行求和，表示每个真实类别的样本总数。
    class_acc = np.divide(correct_per_class, total_per_class, out=np.zeros_like(correct_per_class), where=total_per_class > 0)  # 安全计算每类准确率，避免除以 0。
    return class_acc  # 返回每个类别的准确率数组。


def save_confusion_matrix_csv(cm: np.ndarray, class_names: List[str], csv_path: Union[str, Path]) -> None:  # 定义函数：把混淆矩阵保存成 CSV。
    csv_path = Path(csv_path)  # 把 CSV 保存路径转换成 Path 对象。
    csv_path.parent.mkdir(parents=True, exist_ok=True)  # 确保 CSV 所在目录存在。
    with csv_path.open("w", newline="", encoding="utf-8") as file:  # 以写入模式打开 CSV 文件。
        writer = csv.writer(file)  # 创建 CSV 写入器。
        writer.writerow(["true\\pred"] + class_names)  # 写入表头，第一列表示真实类别，其余列表示预测类别。
        for row_index, class_name in enumerate(class_names):  # 按类别逐行写入混淆矩阵。
            writer.writerow([class_name] + cm[row_index].astype(int).tolist())  # 写入当前真实类别对应的一整行预测计数。


def save_class_accuracy_csv(cm: np.ndarray, class_names: List[str], csv_path: Union[str, Path]) -> None:  # 定义函数：保存每个类别的测试准确率。
    csv_path = Path(csv_path)  # 把 CSV 保存路径转换成 Path 对象。
    csv_path.parent.mkdir(parents=True, exist_ok=True)  # 确保 CSV 所在目录存在。
    class_acc = compute_class_accuracy(cm)  # 根据混淆矩阵计算每类准确率。
    support = np.sum(cm, axis=1)  # 计算每个真实类别在测试集中的样本数。
    with csv_path.open("w", newline="", encoding="utf-8") as file:  # 以写入模式打开 CSV 文件。
        writer = csv.writer(file)  # 创建 CSV 写入器。
        writer.writerow(["class_id", "class_name", "support", "correct", "accuracy"])  # 写入表头。
        for index, class_name in enumerate(class_names):  # 遍历每个类别。
            writer.writerow([index, class_name, int(support[index]), int(cm[index, index]), float(class_acc[index])])  # 写入当前类别的样本数、正确数和准确率。


def save_predictions_csv(samples: List[Sample], y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], csv_path: Union[str, Path]) -> None:  # 定义函数：保存测试集中每张图片的预测结果。
    csv_path = Path(csv_path)  # 把 CSV 保存路径转换成 Path 对象。
    csv_path.parent.mkdir(parents=True, exist_ok=True)  # 确保 CSV 所在目录存在。
    with csv_path.open("w", newline="", encoding="utf-8") as file:  # 以写入模式打开 CSV 文件。
        writer = csv.writer(file)  # 创建 CSV 写入器。
        writer.writerow(["image_path", "true_label", "true_class", "pred_label", "pred_class", "correct"])  # 写入表头。
        for index, sample in enumerate(samples):  # 遍历测试集中的每一个样本。
            image_path, label_id, class_name = sample  # 拆出图片路径、真实标签和真实类别名。
            true_label = int(y_true[index])  # 读取当前样本的真实标签。
            pred_label = int(y_pred[index])  # 读取当前样本的预测标签。
            pred_class = class_names[pred_label]  # 根据预测标签得到预测类别名称。
            correct = bool(true_label == pred_label)  # 判断当前样本是否预测正确。
            writer.writerow([image_path, true_label, class_name, pred_label, pred_class, correct])  # 写入当前样本预测结果。


def save_metrics_json(metrics: Dict[str, object], json_path: Union[str, Path]) -> None:  # 定义函数：把测试指标保存成 JSON 文件。
    json_path = Path(json_path)  # 把 JSON 保存路径转换成 Path 对象。
    json_path.parent.mkdir(parents=True, exist_ok=True)  # 确保 JSON 所在目录存在。
    with json_path.open("w", encoding="utf-8") as file:  # 以写入模式打开 JSON 文件。
        json.dump(metrics, file, ensure_ascii=False, indent=2)  # 把测试指标字典保存成易读 JSON。


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], figure_path: Union[str, Path]) -> None:  # 定义函数：绘制并保存混淆矩阵图片。
    figure_path = Path(figure_path)  # 把图片保存路径转换成 Path 对象。
    figure_path.parent.mkdir(parents=True, exist_ok=True)  # 确保图片所在目录存在。
    fig, ax = plt.subplots(figsize=(10, 8))  # 创建一张图和一个坐标轴。
    image = ax.imshow(cm)  # 使用 imshow 绘制混淆矩阵热力图。
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)  # 添加颜色条，表示计数大小。
    ax.set_title("Confusion Matrix")  # 设置图标题。
    ax.set_xlabel("Predicted label")  # 设置横轴含义为预测类别。
    ax.set_ylabel("True label")  # 设置纵轴含义为真实类别。
    ax.set_xticks(np.arange(len(class_names)))  # 设置横轴刻度位置。
    ax.set_yticks(np.arange(len(class_names)))  # 设置纵轴刻度位置。
    ax.set_xticklabels(class_names, rotation=45, ha="right")  # 设置横轴类别名称，并旋转以避免重叠。
    ax.set_yticklabels(class_names)  # 设置纵轴类别名称。
    for i in range(cm.shape[0]):  # 遍历混淆矩阵的每一行。
        for j in range(cm.shape[1]):  # 遍历混淆矩阵的每一列。
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=7)  # 在每个格子中写入对应计数。
    fig.tight_layout()  # 自动调整图像布局，减少文字被裁剪的情况。
    fig.savefig(figure_path, dpi=200)  # 把混淆矩阵图保存成 PNG 文件。
    plt.close(fig)  # 关闭当前图，释放内存。


def resolve_train_log_path(args: argparse.Namespace, project_dir: Path) -> Optional[Path]:  # 定义函数：确定用于画训练曲线的 train_log.csv 路径。
    if args.train_log.strip() != "":  # 如果在命令行显式提供了 train_log。
        train_log_path = resolve_path(args.train_log, project_dir)  # 把提供的 train_log 路径解析成 Path。
        if not train_log_path.exists():  # 检查该路径是否存在。
            raise FileNotFoundError(f"用户指定的 train_log 不存在：{train_log_path}")  # 如果不存在，就主动报错。
        return train_log_path  # 返回指定的训练日志路径。
    best_hparams_path = resolve_path(args.best_hparams_json, project_dir)  # 解析 best_hyperparams.json 的路径。
    search_dir = resolve_path(args.search_dir, project_dir)  # 解析超参数搜索目录。
    if not best_hparams_path.exists():  # 检查 best_hyperparams.json 是否存在。
        return None  # 如果不存在，就返回 None，表示无法自动画训练曲线。
    best_info = load_json_dict(best_hparams_path)  # 读取 best_hyperparams.json。
    candidate_paths: List[Path] = []  # 创建候选 train_log 路径列表。
    train_log_text = str(best_info.get("train_log_path", ""))  # 尝试从 JSON 中读取 train_log_path 字段。
    if train_log_text != "":  # 如果 JSON 中确实有 train_log_path。
        raw_path = Path(train_log_text)  # 把该路径字符串转换成 Path 对象。
        candidate_paths.append(raw_path)  # 把原始路径加入候选列表。
        candidate_paths.append(project_dir / raw_path)  # 把它也解释成项目根目录下的相对路径并加入候选列表。
    run_id = str(best_info.get("run_id", ""))  # 尝试从 JSON 中读取最佳 run_id。
    if run_id != "":  # 如果存在最佳 run_id。
        candidate_paths.append(search_dir / run_id / "train_log.csv")  # 根据 search_dir 和 run_id 推断该 run 的训练日志路径。
    for candidate in candidate_paths:  # 遍历所有候选路径。
        if candidate.exists():  # 检查当前候选路径是否存在。
            return candidate  # 如果存在，就返回该路径。
    return None  # 如果所有候选路径都不存在，就返回 None。


def read_training_log(log_csv_path: Union[str, Path]) -> Dict[str, List[float]]:  # 定义函数：读取 train_log.csv 中的逐 epoch 指标。
    log_csv_path = Path(log_csv_path)  # 把日志路径转换成 Path 对象。
    if not log_csv_path.exists():  # 检查日志文件是否存在。
        raise FileNotFoundError(f"找不到训练日志：{log_csv_path}")  # 如果不存在，就主动报错。
    epochs: List[float] = []  # 创建列表，用来保存 epoch 编号。
    train_losses: List[float] = []  # 创建列表，用来保存训练 loss。
    val_losses: List[float] = []  # 创建列表，用来保存验证 loss。
    train_accs: List[float] = []  # 创建列表，用来保存训练 accuracy。
    val_accs: List[float] = []  # 创建列表，用来保存验证 accuracy。
    lrs: List[float] = []  # 创建列表，用来保存学习率。
    with log_csv_path.open("r", newline="", encoding="utf-8") as file:  # 以读取模式打开训练日志 CSV。
        reader = csv.DictReader(file)  # 创建字典形式的 CSV 读取器。
        for row in reader:  # 遍历每一个 epoch 的记录。
            epochs.append(float(row["epoch"]))  # 读取并保存 epoch 编号。
            train_loss_key = "train_data_loss" if "train_data_loss" in row and row["train_data_loss"] != "" else "train_loss"  # 优先使用不含 L2 的 train_data_loss；若缺失则使用 train_loss。
            train_losses.append(float(row[train_loss_key]))  # 读取并保存训练 loss。
            val_losses.append(float(row["val_loss"]))  # 读取并保存验证 loss。
            train_accs.append(float(row["train_acc"]))  # 读取并保存训练 accuracy。
            val_accs.append(float(row["val_acc"]))  # 读取并保存验证 accuracy。
            lrs.append(float(row["lr"]))  # 读取并保存学习率。
    log_data = {"epoch": epochs, "train_loss": train_losses, "val_loss": val_losses, "train_acc": train_accs, "val_acc": val_accs, "lr": lrs}  # 把所有曲线数据整理成字典。
    return log_data  # 返回训练日志数据字典。


def plot_loss_curve(log_data: Dict[str, List[float]], figure_path: Union[str, Path]) -> None:  # 定义函数：绘制训练 loss 和验证 loss 曲线。
    figure_path = Path(figure_path)  # 把图片保存路径转换成 Path 对象。
    figure_path.parent.mkdir(parents=True, exist_ok=True)  # 确保图片所在目录存在。
    fig, ax = plt.subplots(figsize=(8, 5))  # 创建图像和坐标轴。
    ax.plot(log_data["epoch"], log_data["train_loss"], label="train_loss")  # 绘制训练 loss 曲线。
    ax.plot(log_data["epoch"], log_data["val_loss"], label="val_loss")  # 绘制验证 loss 曲线。
    ax.set_title("Training and Validation Loss")  # 设置图标题。
    ax.set_xlabel("Epoch")  # 设置横轴为 epoch。
    ax.set_ylabel("Loss")  # 设置纵轴为 loss。
    ax.legend()  # 显示曲线图例。
    ax.grid(True)  # 显示网格线，方便观察趋势。
    fig.tight_layout()  # 自动调整布局。
    fig.savefig(figure_path, dpi=200)  # 保存 loss 曲线图。
    plt.close(fig)  # 关闭图像，释放内存。


def plot_accuracy_curve(log_data: Dict[str, List[float]], figure_path: Union[str, Path]) -> None:  # 定义函数：绘制训练 accuracy 和验证 accuracy 曲线。
    figure_path = Path(figure_path)  # 把图片保存路径转换成 Path 对象。
    figure_path.parent.mkdir(parents=True, exist_ok=True)  # 确保图片所在目录存在。
    fig, ax = plt.subplots(figsize=(8, 5))  # 创建图像和坐标轴。
    ax.plot(log_data["epoch"], log_data["train_acc"], label="train_acc")  # 绘制训练 accuracy 曲线。
    ax.plot(log_data["epoch"], log_data["val_acc"], label="val_acc")  # 绘制验证 accuracy 曲线。
    ax.set_title("Training and Validation Accuracy")  # 设置图标题。
    ax.set_xlabel("Epoch")  # 设置横轴为 epoch。
    ax.set_ylabel("Accuracy")  # 设置纵轴为 accuracy。
    ax.legend()  # 显示曲线图例。
    ax.grid(True)  # 显示网格线，方便观察趋势。
    fig.tight_layout()  # 自动调整布局。
    fig.savefig(figure_path, dpi=200)  # 保存 accuracy 曲线图。
    plt.close(fig)  # 关闭图像，释放内存。


def visualize_first_layer_weights(model: ThreeLayerMLP, num_images: int, figure_path: Union[str, Path]) -> None:  # 定义函数：把第一层隐藏层权重恢复成图片形状并可视化。
    figure_path = Path(figure_path)  # 把图片保存路径转换成 Path 对象。
    figure_path.parent.mkdir(parents=True, exist_ok=True)  # 确保图片所在目录存在。
    W1 = model.fc1.W  # 取出第一层权重矩阵，shape 为 input_dim×hidden_dim1。
    input_dim = IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3  # 计算一张图片展平后的输入维度。
    if W1.shape[0] != input_dim:  # 检查第一层权重的输入维度是否等于 64×64×3。
        raise ValueError(f"第一层权重无法恢复成图片：W1.shape[0]={W1.shape[0]}，但期望 {input_dim}")  # 如果维度不匹配，就主动报错。
    num_units = min(int(num_images), W1.shape[1])  # 计算实际要展示的隐藏单元数量，不能超过隐藏层宽度。
    unit_indices = np.linspace(0, W1.shape[1] - 1, num_units, dtype=int)  # 在所有隐藏单元中均匀选择若干个用于可视化。
    ncols = int(math.ceil(math.sqrt(num_units)))  # 根据展示数量自动计算网格列数。
    nrows = int(math.ceil(num_units / ncols))  # 根据展示数量自动计算网格行数。
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))  # 创建权重可视化网格。
    axes_array = np.asarray(axes).reshape(-1)  # 把 axes 转换成一维数组，方便循环访问。
    for plot_index, unit_index in enumerate(unit_indices):  # 遍历每一个要展示的隐藏单元。
        ax = axes_array[plot_index]  # 取出当前子图坐标轴。
        weight_vector = W1[:, int(unit_index)]  # 取出某个隐藏单元对应的输入权重列。
        weight_image = weight_vector.reshape(IMAGE_SIZE[1], IMAGE_SIZE[0], 3)  # 把权重向量恢复成 H×W×3 图片形状。
        w_min = float(weight_image.min())  # 计算当前权重图中的最小值。
        w_max = float(weight_image.max())  # 计算当前权重图中的最大值。
        weight_vis = (weight_image - w_min) / (w_max - w_min + 1e-8)  # 把权重线性归一化到 0 到 1，方便作为 RGB 图片显示。
        ax.imshow(np.clip(weight_vis, 0.0, 1.0))  # 显示当前隐藏单元的权重图。
        ax.set_title(f"unit {int(unit_index)}", fontsize=9)  # 设置当前子图标题。
        ax.axis("off")  # 关闭坐标轴刻度。
    for empty_index in range(num_units, len(axes_array)):  # 遍历没有使用到的空子图。
        axes_array[empty_index].axis("off")  # 关闭空子图坐标轴。
    fig.suptitle("First Layer Weight Visualization", fontsize=14)  # 设置整张图的总标题。
    fig.tight_layout()  # 自动调整布局。
    fig.savefig(figure_path, dpi=200)  # 保存第一层权重可视化图片。
    plt.close(fig)  # 关闭图像，释放内存。


def save_error_examples_csv(samples: List[Sample], wrong_indices: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], csv_path: Union[str, Path]) -> None:  # 定义函数：保存所有错例信息到 CSV。
    csv_path = Path(csv_path)  # 把 CSV 保存路径转换成 Path 对象。
    csv_path.parent.mkdir(parents=True, exist_ok=True)  # 确保 CSV 所在目录存在。
    with csv_path.open("w", newline="", encoding="utf-8") as file:  # 以写入模式打开 CSV 文件。
        writer = csv.writer(file)  # 创建 CSV 写入器。
        writer.writerow(["image_path", "true_label", "true_class", "pred_label", "pred_class"])  # 写入表头。
        for index in wrong_indices:  # 遍历每一个预测错误的样本下标。
            sample = samples[int(index)]  # 取出当前错例样本。
            image_path, label_id, class_name = sample  # 拆出图片路径、真实标签和真实类别名。
            pred_label = int(y_pred[int(index)])  # 读取当前错例的预测标签。
            pred_class = class_names[pred_label]  # 根据预测标签得到预测类别名。
            writer.writerow([image_path, int(y_true[int(index)]), class_name, pred_label, pred_class])  # 写入当前错例记录。


def plot_error_examples(samples: List[Sample], y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], num_examples: int, seed: int, figure_path: Union[str, Path]) -> None:  # 定义函数：绘制若干测试集错例图片。
    figure_path = Path(figure_path)  # 把图片保存路径转换成 Path 对象。
    figure_path.parent.mkdir(parents=True, exist_ok=True)  # 确保图片所在目录存在。
    wrong_indices = np.where(y_true != y_pred)[0]  # 找出所有预测错误样本的下标。
    rng = np.random.default_rng(seed)  # 创建随机数生成器，用来随机抽取错例。
    if len(wrong_indices) > num_examples:  # 如果错例数量超过希望展示的数量。
        selected_indices = rng.choice(wrong_indices, size=num_examples, replace=False)  # 随机抽取 num_examples 个错例。
    else:  # 如果错例数量不足或刚好够。
        selected_indices = wrong_indices  # 直接展示全部错例。
    n_show = max(1, len(selected_indices))  # 计算实际展示数量；至少为 1，方便无错例时也生成图片。
    ncols = min(4, n_show)  # 设置错例图最多 4 列。
    nrows = int(math.ceil(n_show / ncols))  # 根据错例数量计算行数。
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))  # 创建错例可视化网格。
    axes_array = np.asarray(axes).reshape(-1)  # 把 axes 转换成一维数组，方便循环访问。
    if len(selected_indices) == 0:  # 如果测试集中没有预测错误的样本。
        axes_array[0].text(0.5, 0.5, "No error examples found", ha="center", va="center")  # 在图中写入没有错例的提示。
        axes_array[0].axis("off")  # 关闭坐标轴。
    for plot_index, sample_index in enumerate(selected_indices):  # 遍历被选中的错例样本。
        ax = axes_array[plot_index]  # 取出当前子图坐标轴。
        image_path, label_id, class_name = samples[int(sample_index)]  # 读取当前错例的图片路径和真实类别。
        image = load_image(image_path, image_size=IMAGE_SIZE, normalize=True)  # 读取原图并缩放到 0 到 1，方便显示。
        true_label = int(y_true[int(sample_index)])  # 读取当前错例的真实标签。
        pred_label = int(y_pred[int(sample_index)])  # 读取当前错例的预测标签。
        true_class = class_names[true_label]  # 根据真实标签得到真实类别名称。
        pred_class = class_names[pred_label]  # 根据预测标签得到预测类别名称。
        ax.imshow(np.clip(image, 0.0, 1.0))  # 显示当前错例图片。
        ax.set_title(f"True: {true_class}\nPred: {pred_class}", fontsize=9)  # 设置标题，显示真实类别和预测类别。
        ax.axis("off")  # 关闭坐标轴刻度。
    for empty_index in range(len(selected_indices), len(axes_array)):  # 遍历没有使用到的空子图。
        axes_array[empty_index].axis("off")  # 关闭空子图坐标轴。
    fig.suptitle("Error Examples", fontsize=14)  # 设置整张图的总标题。
    fig.tight_layout()  # 自动调整布局。
    fig.savefig(figure_path, dpi=200)  # 保存错例图。
    plt.close(fig)  # 关闭图像，释放内存。


def main() -> None:  # 定义主函数：组织测试评估和可视化的完整流程。
    args = parse_args()  # 读取命令行参数。
    if args.batch_size <= 0:  # 检查 batch_size 是否为正整数。
        raise ValueError(f"batch_size 必须为正整数，但当前是 {args.batch_size}")  # 如果 batch_size 不合法，就主动报错。
    project_dir = Path(__file__).resolve().parent  # 获取当前 evaluate.py 所在目录，也就是项目根目录。
    data_dir = resolve_path(args.data_dir, project_dir)  # 解析 EuroSAT_RGB 数据集根目录。
    test_csv = resolve_path(args.test_csv, project_dir)  # 解析测试集 test.csv 路径。
    weights_path = resolve_path(args.weights, project_dir)  # 解析模型权重路径。
    normalization_json = resolve_path(args.normalization_json, project_dir)  # 解析 normalization.json 路径。
    output_dir = resolve_path(args.output_dir, project_dir)  # 解析评估输出总目录。
    results_dir = output_dir / "results"  # 设置数值结果输出目录。
    figures_dir = output_dir / "figures"  # 设置图片结果输出目录。
    results_dir.mkdir(parents=True, exist_ok=True)  # 创建 results 输出目录。
    figures_dir.mkdir(parents=True, exist_ok=True)  # 创建 figures 输出目录。
    if not data_dir.exists():  # 检查数据集根目录是否存在。
        raise FileNotFoundError(f"数据集目录不存在：{data_dir}")  # 如果不存在，就主动报错。
    if not test_csv.exists():  # 检查测试集 CSV 是否存在。
        raise FileNotFoundError(f"测试集 CSV 不存在：{test_csv}，请先运行 data_utils.py 生成划分文件。")  # 如果不存在，就主动报错。
    model, metadata, mean_rgb, std_rgb = load_model_from_npz(weights_path, normalization_json)  # 加载模型权重、元信息和标准化参数。
    test_samples = read_split_csv(test_csv, data_dir)  # 从 test.csv 读取测试集样本列表。
    if len(test_samples) == 0:  # 检查测试集是否为空。
        raise ValueError("测试集为空，无法进行测试评估。")  # 如果测试集为空，就主动报错。
    class_names = get_class_names(metadata, test_samples, model.output_dim)  # 获取类别编号对应的类别名称。
    y_true, y_pred, test_loss, test_accuracy = predict_dataset(model, test_samples, args.batch_size, mean_rgb, std_rgb)  # 在测试集上预测并计算 loss/accuracy。
    cm = build_confusion_matrix(y_true, y_pred, model.output_dim)  # 根据真实标签和预测标签构造混淆矩阵。
    class_acc = compute_class_accuracy(cm)  # 根据混淆矩阵计算每个类别的测试准确率。
    confusion_csv_path = results_dir / "confusion_matrix.csv"  # 设置混淆矩阵 CSV 保存路径。
    class_acc_csv_path = results_dir / "class_accuracy.csv"  # 设置各类别准确率 CSV 保存路径。
    predictions_csv_path = results_dir / "test_predictions.csv"  # 设置逐样本预测结果 CSV 保存路径。
    error_csv_path = results_dir / "error_examples.csv"  # 设置错例 CSV 保存路径。
    metrics_json_path = results_dir / "test_metrics.json"  # 设置测试指标 JSON 保存路径。
    confusion_png_path = figures_dir / "confusion_matrix.png"  # 设置混淆矩阵图片保存路径。
    loss_curve_png_path = figures_dir / "loss_curve.png"  # 设置 loss 曲线图片保存路径。
    accuracy_curve_png_path = figures_dir / "accuracy_curve.png"  # 设置 accuracy 曲线图片保存路径。
    first_layer_png_path = figures_dir / "first_layer_weights.png"  # 设置第一层权重可视化图片保存路径。
    error_examples_png_path = figures_dir / "error_examples.png"  # 设置错例图片保存路径。
    wrong_indices = np.where(y_true != y_pred)[0]  # 找出所有测试集错例下标。
    metrics = {  # 创建测试指标字典。
        "weights_path": str(weights_path),  # 记录被测试的权重文件路径。
        "test_csv": str(test_csv),  # 记录测试集 CSV 路径。
        "num_test_samples": int(len(test_samples)),  # 记录测试集样本数量。
        "test_loss": float(test_loss),  # 记录测试集平均交叉熵 loss。
        "test_accuracy": float(test_accuracy),  # 记录测试集整体 accuracy。
        "num_errors": int(len(wrong_indices)),  # 记录测试集错误分类样本数量。
        "class_names": class_names,  # 记录类别名称顺序。
        "class_accuracy": {class_names[i]: float(class_acc[i]) for i in range(len(class_names))},  # 记录每个类别的准确率。
        "metadata": metadata,  # 记录模型训练时保存的元信息。
    }  # 测试指标字典创建完毕。
    save_confusion_matrix_csv(cm, class_names, confusion_csv_path)  # 保存混淆矩阵 CSV。
    save_class_accuracy_csv(cm, class_names, class_acc_csv_path)  # 保存每个类别准确率 CSV。
    save_predictions_csv(test_samples, y_true, y_pred, class_names, predictions_csv_path)  # 保存每张测试图片的预测结果。
    save_error_examples_csv(test_samples, wrong_indices, y_true, y_pred, class_names, error_csv_path)  # 保存所有错例信息。
    save_metrics_json(metrics, metrics_json_path)  # 保存测试指标 JSON。
    plot_confusion_matrix(cm, class_names, confusion_png_path)  # 绘制并保存混淆矩阵图片。
    visualize_first_layer_weights(model, args.num_weight_images, first_layer_png_path)  # 绘制并保存第一层权重可视化图片。
    plot_error_examples(test_samples, y_true, y_pred, class_names, args.num_error_examples, args.seed, error_examples_png_path)  # 绘制并保存错例图。
    train_log_path = resolve_train_log_path(args, project_dir)  # 尝试自动找到最佳 run 的 train_log.csv。
    if train_log_path is not None:  # 如果找到了训练日志。
        log_data = read_training_log(train_log_path)  # 读取训练日志中的逐 epoch 指标。
        plot_loss_curve(log_data, loss_curve_png_path)  # 绘制并保存训练/验证 loss 曲线。
        plot_accuracy_curve(log_data, accuracy_curve_png_path)  # 绘制并保存训练/验证 accuracy 曲线。
        print("训练曲线使用的日志：", train_log_path)  # 打印训练日志路径，方便核对。
    else:  # 如果没有找到训练日志。
        print("未找到 train_log.csv，因此跳过 loss/accuracy 曲线绘制。")  # 打印跳过曲线绘制的提示。
    print("Test Accuracy:", test_accuracy)  # 打印测试集整体 accuracy。
    print("Test Loss:", test_loss)  # 打印测试集平均 loss。
    print("Class order:", class_names)  # 打印混淆矩阵对应的类别顺序。
    print("Confusion Matrix:")  # 打印混淆矩阵标题。
    print(cm)  # 打印混淆矩阵本身，行是真实类别，列是预测类别。
    print("混淆矩阵 CSV：", confusion_csv_path)  # 打印混淆矩阵 CSV 保存路径。
    print("混淆矩阵图片：", confusion_png_path)  # 打印混淆矩阵图片保存路径。
    print("类别准确率 CSV：", class_acc_csv_path)  # 打印类别准确率 CSV 保存路径。
    print("测试指标 JSON：", metrics_json_path)  # 打印测试指标 JSON 保存路径。
    print("逐样本预测结果 CSV：", predictions_csv_path)  # 打印逐样本预测结果 CSV 保存路径。
    print("第一层权重可视化：", first_layer_png_path)  # 打印第一层权重可视化图片路径。
    print("错例 CSV：", error_csv_path)  # 打印错例 CSV 保存路径。
    print("错例图片：", error_examples_png_path)  # 打印错例图片保存路径。


if __name__ == "__main__":  # 只有直接运行 evaluate.py 时，才执行主流程。
    main()  # 调用主函数，开始测试评估和可视化。
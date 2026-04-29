from pathlib import Path  # 导入 Path，用来处理项目路径、输出路径、每个实验 run 的保存路径。
from typing import Dict, List, Optional, Tuple, Union  # 导入类型标注工具，方便说明函数输入输出的数据类型。
import argparse  # 导入 argparse，用来从命令行读取超参数搜索配置。
import csv  # 导入 csv，用来把每组超参数组合的结果保存成 CSV 文件。
import json  # 导入 json，用来保存最优超参数组合的 summary 文件。
import shutil  # 导入 shutil，用来复制整体最优模型权重文件。
import time  # 导入 time，用来统计每组超参数实验耗时。
import numpy as np  # 导入 NumPy，使用 NumPy 进行矩阵运算和随机抽样。

from data_utils import IMAGE_SIZE  # 从 data_utils.py 导入统一图像尺寸，例如 (64, 64)。
from data_utils import read_split_csv  # 从 data_utils.py 导入读取 train.csv / val.csv 的函数。
from mlp_numpy import ThreeLayerMLP  # 从 mlp_numpy.py 导入真正的三层 MLP 模型。
from mlp_numpy import load_normalization_params  # 从 mlp_numpy.py 导入读取训练集 RGB 均值和标准差的函数。
from train import SGD  # 从 train.py 导入已经实现好的 SGD 优化器。
from train import append_log_row  # 从 train.py 导入向训练日志 CSV 追加一行记录的函数。
from train import check_training_files  # 从 train.py 导入检查训练所需文件是否存在的函数。
from train import evaluate  # 从 train.py 导入在验证集上评估模型的函数。
from train import get_current_lr  # 从 train.py 导入 step learning rate decay 的学习率计算函数。
from train import load_json_dict  # 从 train.py 导入读取 JSON 文件的函数。
from train import resolve_path  # 从 train.py 导入把相对路径转换成项目内绝对路径的函数。
from train import save_model_npz  # 从 train.py 导入保存模型权重和元信息的函数。
from train import train_one_epoch  # 从 train.py 导入训练一个 epoch 的函数。
from train import write_log_header  # 从 train.py 导入创建训练日志 CSV 表头的函数。


def parse_args() -> argparse.Namespace:  # 定义函数：解析命令行参数。
    parser = argparse.ArgumentParser(description="Hyperparameter search for NumPy three-layer MLP on EuroSAT_RGB.")  # 创建命令行参数解析器。
    parser.add_argument("--data_dir", type=str, default="data/EuroSAT_RGB", help="EuroSAT_RGB 数据集根目录。")  # 添加数据集根目录参数。
    parser.add_argument("--train_csv", type=str, default="outputs/splits/train.csv", help="训练集划分 CSV 路径。")  # 添加训练集 CSV 路径参数。
    parser.add_argument("--val_csv", type=str, default="outputs/splits/val.csv", help="验证集划分 CSV 路径。")  # 添加验证集 CSV 路径参数。
    parser.add_argument("--normalization_json", type=str, default="outputs/normalization.json", help="训练集 RGB 均值和标准差 JSON 路径。")  # 添加归一化参数 JSON 路径。
    parser.add_argument("--search_dir", type=str, default="outputs/hparam_search", help="超参数搜索结果保存目录。")  # 添加超参数搜索输出目录。
    parser.add_argument("--results_csv", type=str, default="outputs/results/hyperparam_results.csv", help="超参数搜索汇总 CSV 保存路径。")  # 添加超参数搜索汇总结果 CSV 路径。
    parser.add_argument("--search_mode", type=str, default="random", choices=["grid", "random"], help="搜索方式：grid 表示网格搜索，random 表示随机搜索。")  # 添加搜索方式参数。
    parser.add_argument("--max_trials", type=int, default=8, help="最多运行多少组组合；random 模式常用，grid 模式下 0 表示运行全部组合。")  # 添加最多实验次数参数。
    parser.add_argument("--learning_rates", type=str, default="0.003,0.01,0.03", help="待搜索学习率列表，用逗号分隔。")  # 添加学习率候选列表。
    parser.add_argument("--hidden_dims", type=str, default="64,128,256", help="待搜索隐藏层大小列表，用逗号分隔；两个隐藏层默认使用相同 hidden_dim。")  # 添加隐藏层大小候选列表。
    parser.add_argument("--weight_decays", type=str, default="0.0,0.0001,0.001", help="待搜索 L2 正则化强度列表，用逗号分隔。")  # 添加 weight decay 候选列表。
    parser.add_argument("--activations", type=str, default="relu,tanh", help="待搜索激活函数列表，用逗号分隔，可选 relu、tanh、sigmoid。")  # 添加激活函数候选列表。
    parser.add_argument("--batch_size", type=int, default=64, help="每组实验使用的 mini-batch 大小。")  # 添加 batch size 参数。
    parser.add_argument("--epochs", type=int, default=10, help="每组超参数组合训练的 epoch 数；搜索阶段建议先用较小值。")  # 添加每组实验训练 epoch 数。
    parser.add_argument("--decay_rate", type=float, default=0.9, help="学习率 step decay 的衰减倍率。")  # 添加学习率衰减倍率。
    parser.add_argument("--decay_every", type=int, default=5, help="每隔多少个 epoch 衰减一次学习率。")  # 添加学习率衰减间隔。
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于随机搜索、模型初始化和训练集 shuffle。")  # 添加随机种子参数。
    return parser.parse_args()  # 返回解析后的命令行参数。


def parse_float_list(text: str) -> List[float]:  # 定义函数：把形如 "0.003,0.01" 的字符串解析成 float 列表。
    values: List[float] = []  # 创建空列表，用来存放解析出的浮点数。
    for item in text.split(","):  # 按逗号切分字符串，逐个读取候选值。
        stripped = item.strip()  # 去掉当前候选值前后的空格。
        if stripped == "":  # 判断当前候选值是否是空字符串。
            continue  # 如果是空字符串，就跳过它。
        values.append(float(stripped))  # 把当前候选值转换成 float 并加入列表。
    if len(values) == 0:  # 检查最终是否至少解析出一个数。
        raise ValueError(f"无法从字符串中解析出任何 float：{text}")  # 如果没有任何候选值，就主动报错。
    return values  # 返回解析出的 float 列表。


def parse_int_list(text: str) -> List[int]:  # 定义函数：把形如 "64,128,256" 的字符串解析成 int 列表。
    values: List[int] = []  # 创建空列表，用来存放解析出的整数。
    for item in text.split(","):  # 按逗号切分字符串，逐个读取候选值。
        stripped = item.strip()  # 去掉当前候选值前后的空格。
        if stripped == "":  # 判断当前候选值是否是空字符串。
            continue  # 如果是空字符串，就跳过它。
        values.append(int(stripped))  # 把当前候选值转换成 int 并加入列表。
    if len(values) == 0:  # 检查最终是否至少解析出一个整数。
        raise ValueError(f"无法从字符串中解析出任何 int：{text}")  # 如果没有任何候选值，就主动报错。
    return values  # 返回解析出的 int 列表。


def parse_str_list(text: str) -> List[str]:  # 定义函数：把形如 "relu,tanh" 的字符串解析成字符串列表。
    values: List[str] = []  # 创建空列表，用来存放解析出的字符串。
    for item in text.split(","):  # 按逗号切分字符串，逐个读取候选值。
        stripped = item.strip().lower()  # 去掉空格并转换成小写，保证 "ReLU" 和 "relu" 都能识别。
        if stripped == "":  # 判断当前候选值是否为空字符串。
            continue  # 如果是空字符串，就跳过它。
        values.append(stripped)  # 把当前候选值加入字符串列表。
    if len(values) == 0:  # 检查最终是否至少解析出一个字符串。
        raise ValueError(f"无法从字符串中解析出任何字符串：{text}")  # 如果没有任何候选值，就主动报错。
    return values  # 返回解析出的字符串列表。


def validate_activation_names(activations: List[str]) -> None:  # 定义函数：检查激活函数名称是否合法。
    allowed = {"relu", "tanh", "sigmoid"}  # 定义本项目支持的激活函数名称集合。
    for activation in activations:  # 遍历输入的每一个激活函数名称。
        if activation not in allowed:  # 判断当前激活函数是否不在允许集合里。
            raise ValueError(f"不支持的激活函数：{activation}，可选值为 relu、tanh、sigmoid。")  # 如果不合法，就主动报错。


def format_float_for_name(value: float) -> str:  # 定义函数：把浮点数转换成适合文件夹名字的字符串。
    text = f"{value:g}"  # 用紧凑格式把浮点数转换成字符串，例如 0.001。
    text = text.replace("-", "m")  # 把负号替换成 m，避免文件名里出现特殊含义。
    text = text.replace(".", "p")  # 把小数点替换成 p，避免文件名过于混乱。
    return text  # 返回适合放进文件夹名字的字符串。


def build_search_grid(learning_rates: List[float], hidden_dims: List[int], weight_decays: List[float], activations: List[str]) -> List[Dict[str, object]]:  # 定义函数：根据候选列表构造完整网格。
    configs: List[Dict[str, object]] = []  # 创建空列表，用来存放所有超参数组合。
    for lr in learning_rates:  # 遍历所有候选学习率。
        for hidden_dim in hidden_dims:  # 遍历所有候选隐藏层大小。
            for weight_decay in weight_decays:  # 遍历所有候选 L2 正则化强度。
                for activation in activations:  # 遍历所有候选激活函数。
                    config = {  # 创建当前这一组超参数组合字典。
                        "lr": float(lr),  # 保存当前学习率。
                        "hidden_dim": int(hidden_dim),  # 保存当前隐藏层大小。
                        "weight_decay": float(weight_decay),  # 保存当前 L2 正则化强度。
                        "activation": str(activation),  # 保存当前激活函数名称。
                    }  # 当前超参数组合字典创建完毕。
                    configs.append(config)  # 把当前超参数组合加入总列表。
    return configs  # 返回完整的超参数组合列表。


def select_search_configs(configs: List[Dict[str, object]], search_mode: str, max_trials: int, seed: int) -> List[Dict[str, object]]:  # 定义函数：根据搜索模式选择实际要跑的组合。
    if len(configs) == 0:  # 检查候选组合是否为空。
        raise ValueError("超参数组合列表为空，无法进行搜索。")  # 如果为空，就主动报错。
    if search_mode == "grid":  # 判断是否使用网格搜索。
        if max_trials <= 0:  # 判断是否要求运行完整网格。
            return configs  # 如果 max_trials 小于等于 0，就返回全部组合。
        return configs[:min(max_trials, len(configs))]  # 如果 max_trials 为正数，就只运行前 max_trials 组组合。
    if search_mode == "random":  # 判断是否使用随机搜索。
        if max_trials <= 0:  # 检查随机搜索的最大次数是否合法。
            raise ValueError("random 模式下 max_trials 必须为正整数。")  # random 模式下 max_trials 不能小于等于 0。
        rng = np.random.default_rng(seed)  # 创建随机数生成器，保证随机搜索组合可复现。
        indices = rng.permutation(len(configs))  # 对完整组合索引做随机排列。
        selected_indices = indices[:min(max_trials, len(configs))]  # 取前 max_trials 个随机索引。
        selected_configs = [configs[int(index)] for index in selected_indices]  # 根据随机索引取出实际要运行的组合。
        return selected_configs  # 返回随机选择出的组合列表。
    raise ValueError(f"未知 search_mode：{search_mode}")  # 如果搜索模式既不是 grid 也不是 random，就主动报错。


def get_search_result_fieldnames() -> List[str]:  # 定义函数：返回超参数搜索汇总 CSV 的列名。
    fieldnames = [  # 创建 CSV 列名列表。
        "run_id",  # 记录实验编号。
        "status",  # 记录实验状态，ok 表示成功，failed 表示失败。
        "error_message",  # 记录失败原因，成功时为空字符串。
        "lr",  # 记录学习率。
        "hidden_dim",  # 记录第一个隐藏层大小。
        "hidden_dim2",  # 记录第二个隐藏层大小。
        "weight_decay",  # 记录 L2 正则化强度。
        "activation",  # 记录激活函数。
        "batch_size",  # 记录 batch size。
        "epochs",  # 记录本组实验训练 epoch 数。
        "decay_rate",  # 记录学习率衰减倍率。
        "decay_every",  # 记录学习率衰减间隔。
        "best_val_acc",  # 记录本组实验最佳验证集准确率。
        "best_val_loss",  # 记录本组实验最佳验证集 loss。
        "best_train_acc",  # 记录最佳验证集准确率对应 epoch 的训练准确率。
        "best_train_loss",  # 记录最佳验证集准确率对应 epoch 的训练 loss。
        "best_epoch",  # 记录本组实验最佳 epoch。
        "checkpoint_path",  # 记录本组实验 best_model.npz 路径。
        "train_log_path",  # 记录本组实验 train_log.csv 路径。
        "elapsed_seconds",  # 记录本组实验耗时秒数。
    ]  # CSV 列名列表创建完毕。
    return fieldnames  # 返回 CSV 列名列表。


def write_search_header(results_csv_path: Union[str, Path]) -> None:  # 定义函数：创建超参数搜索汇总 CSV 并写入表头。
    results_csv_path = Path(results_csv_path)  # 把结果 CSV 路径转换成 Path 对象。
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)  # 确保结果 CSV 所在目录存在。
    with results_csv_path.open("w", newline="", encoding="utf-8") as file:  # 以写入模式打开结果 CSV。
        writer = csv.DictWriter(file, fieldnames=get_search_result_fieldnames())  # 创建字典形式 CSV 写入器。
        writer.writeheader()  # 写入 CSV 表头。


def append_search_result(results_csv_path: Union[str, Path], result: Dict[str, object]) -> None:  # 定义函数：向超参数搜索汇总 CSV 追加一行结果。
    results_csv_path = Path(results_csv_path)  # 把结果 CSV 路径转换成 Path 对象。
    fieldnames = get_search_result_fieldnames()  # 读取固定的 CSV 列名列表。
    row = {field: result.get(field, "") for field in fieldnames}  # 按固定列名从 result 字典中取值，缺失字段用空字符串。
    with results_csv_path.open("a", newline="", encoding="utf-8") as file:  # 以追加模式打开结果 CSV。
        writer = csv.DictWriter(file, fieldnames=fieldnames)  # 创建字典形式 CSV 写入器。
        writer.writerow(row)  # 写入当前这一组实验结果。


def make_run_name(run_id: int, config: Dict[str, object]) -> str:  # 定义函数：为每组超参数组合生成一个易读的文件夹名称。
    lr_text = format_float_for_name(float(config["lr"]))  # 把学习率转换成适合文件名的字符串。
    wd_text = format_float_for_name(float(config["weight_decay"]))  # 把 weight decay 转换成适合文件名的字符串。
    hidden_text = str(int(config["hidden_dim"]))  # 把 hidden_dim 转换成字符串。
    activation_text = str(config["activation"])  # 把 activation 转换成字符串。
    run_name = f"run_{run_id:03d}_lr{lr_text}_h{hidden_text}_wd{wd_text}_{activation_text}"  # 拼接出当前 run 的文件夹名称。
    return run_name  # 返回当前 run 的文件夹名称。


def infer_output_dim(normalization_params: Dict[str, object]) -> int:  # 定义函数：从 normalization.json 中推断类别数。
    class_to_idx = normalization_params.get("class_to_idx", {})  # 尝试读取类别名到整数标签的映射。
    if isinstance(class_to_idx, dict) and len(class_to_idx) > 0:  # 判断类别映射是否是非空字典。
        return int(len(class_to_idx))  # 如果映射存在，就把类别数设为映射长度。
    return 10  # 如果映射缺失，就默认 EuroSAT 是 10 类。


def run_single_config(run_id: int, config: Dict[str, object], args: argparse.Namespace, data_dir: Path, train_samples: List[Tuple[str, int, str]], val_samples: List[Tuple[str, int, str]], mean_rgb: np.ndarray, std_rgb: np.ndarray, normalization_params: Dict[str, object], search_dir: Path) -> Dict[str, object]:  # 定义函数：训练并评估单组超参数组合。
    start_time = time.time()  # 记录当前实验开始时间。
    run_name = make_run_name(run_id, config)  # 根据 run_id 和超参数组合生成当前实验名称。
    run_dir = search_dir / run_name  # 设置当前实验的输出目录。
    run_dir.mkdir(parents=True, exist_ok=True)  # 创建当前实验输出目录。
    train_log_path = run_dir / "train_log.csv"  # 设置当前实验训练日志保存路径。
    checkpoint_path = run_dir / "best_model.npz"  # 设置当前实验最佳模型保存路径。
    input_dim = int(IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3)  # 根据图片尺寸计算 MLP 输入维度，即 64×64×3。
    hidden_dim = int(config["hidden_dim"])  # 从 config 中读取隐藏层大小。
    hidden_dim2 = hidden_dim  # 两个隐藏层默认使用相同 hidden_dim，符合 Hidden Dimension 单数设置。
    output_dim = infer_output_dim(normalization_params)  # 从类别映射推断输出类别数。
    activation = str(config["activation"])  # 从 config 中读取激活函数名称。
    lr = float(config["lr"])  # 从 config 中读取学习率。
    weight_decay = float(config["weight_decay"])  # 从 config 中读取 L2 正则化强度。
    model_seed = int(args.seed + run_id)  # 为当前实验构造模型随机种子，使不同 run 初始化不同且可复现。
    shuffle_seed = int(args.seed + 10000 + run_id)  # 为当前实验构造训练集 shuffle 随机种子。
    model = ThreeLayerMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, activation=activation, seed=model_seed)  # 初始化当前实验的三层 MLP。
    optimizer = SGD(lr=lr, weight_decay=weight_decay)  # 初始化当前实验的 SGD 优化器。
    train_rng = np.random.default_rng(shuffle_seed)  # 初始化当前实验的训练集 shuffle 随机数生成器。
    write_log_header(train_log_path)  # 为当前实验创建 train_log.csv 并写入表头。
    best_val_acc = -1.0  # 初始化最佳验证集准确率。
    best_val_loss = float("inf")  # 初始化最佳验证集 loss。
    best_train_acc = 0.0  # 初始化最佳 epoch 对应训练准确率。
    best_train_loss = float("inf")  # 初始化最佳 epoch 对应训练 loss。
    best_epoch = 0  # 初始化最佳 epoch 编号。
    print("=" * 100)  # 打印分隔线，方便在终端中区分不同 run。
    print(f"Run {run_id:03d} | lr={lr} | hidden_dim={hidden_dim} | weight_decay={weight_decay} | activation={activation}")  # 打印当前实验的超参数组合。
    for epoch_index in range(args.epochs):  # 按 epoch 循环训练当前超参数组合。
        epoch = epoch_index + 1  # 把从 0 开始的 epoch_index 转换成从 1 开始的 epoch 编号。
        current_lr = get_current_lr(lr, args.decay_rate, args.decay_every, epoch_index)  # 根据 step decay 计算当前 epoch 学习率。
        optimizer.set_lr(current_lr)  # 把优化器当前学习率更新为 current_lr。
        train_loss, train_data_loss, train_acc = train_one_epoch(model, optimizer, train_samples, args.batch_size, mean_rgb, std_rgb, train_rng)  # 在训练集上训练一个 epoch。
        val_loss, val_acc = evaluate(model, val_samples, args.batch_size, mean_rgb, std_rgb)  # 在验证集上评估当前模型。
        append_log_row(train_log_path, epoch, train_loss, train_data_loss, val_loss, train_acc, val_acc, current_lr)  # 把当前 epoch 的训练和验证指标写入当前实验日志。
        print(f"Run {run_id:03d} | Epoch {epoch:03d}/{args.epochs:03d} | lr={current_lr:.6g} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | val_loss={val_loss:.6f}")  # 打印当前 epoch 的关键指标。
        if val_acc > best_val_acc:  # 判断当前验证集准确率是否超过当前 run 的历史最好结果。
            best_val_acc = float(val_acc)  # 更新当前 run 的最佳验证集准确率。
            best_val_loss = float(val_loss)  # 记录最佳验证集准确率对应的验证集 loss。
            best_train_acc = float(train_acc)  # 记录最佳验证集准确率对应的训练准确率。
            best_train_loss = float(train_loss)  # 记录最佳验证集准确率对应的训练 loss。
            best_epoch = int(epoch)  # 记录最佳验证集准确率对应的 epoch 编号。
            metadata = {  # 创建保存模型时需要写入的元信息字典。
                "run_id": run_id,  # 记录当前实验编号。
                "run_name": run_name,  # 记录当前实验名称。
                "input_dim": input_dim,  # 记录输入维度。
                "hidden_dim1": hidden_dim,  # 记录第一个隐藏层维度。
                "hidden_dim2": hidden_dim2,  # 记录第二个隐藏层维度。
                "output_dim": output_dim,  # 记录输出类别数。
                "activation": activation,  # 记录激活函数。
                "best_epoch": best_epoch,  # 记录最佳 epoch。
                "best_val_acc": best_val_acc,  # 记录最佳验证集准确率。
                "best_val_loss": best_val_loss,  # 记录最佳验证集 loss。
                "best_train_acc": best_train_acc,  # 记录对应训练准确率。
                "best_train_loss": best_train_loss,  # 记录对应训练 loss。
                "batch_size": args.batch_size,  # 记录 batch size。
                "epochs": args.epochs,  # 记录搜索阶段 epoch 数。
                "initial_lr": lr,  # 记录初始学习率。
                "current_lr": current_lr,  # 记录保存时的当前学习率。
                "weight_decay": weight_decay,  # 记录 L2 正则化强度。
                "decay_rate": args.decay_rate,  # 记录学习率衰减倍率。
                "decay_every": args.decay_every,  # 记录学习率衰减间隔。
                "seed": model_seed,  # 记录模型初始化种子。
                "shuffle_seed": shuffle_seed,  # 记录 shuffle 种子。
                "class_to_idx": normalization_params.get("class_to_idx", {}),  # 保存类别名称到标签编号的映射。
                "idx_to_class": normalization_params.get("idx_to_class", {}),  # 保存标签编号到类别名称的映射。
                "image_size": list(IMAGE_SIZE),  # 保存图片尺寸。
            }  # 模型元信息字典创建完毕。
            save_model_npz(model, checkpoint_path, metadata, mean_rgb, std_rgb)  # 保存当前 run 的 best model。
            print(f"  Run {run_id:03d} 保存新的 best model：epoch={best_epoch}, val_acc={best_val_acc:.4f}")  # 打印保存 best model 的提示。
    elapsed_seconds = time.time() - start_time  # 计算当前 run 总耗时秒数。
    result = {  # 创建当前 run 的汇总结果字典。
        "run_id": run_name,  # 保存当前 run 的名称。
        "status": "ok",  # 标记当前 run 成功完成。
        "error_message": "",  # 成功时错误信息为空。
        "lr": lr,  # 保存学习率。
        "hidden_dim": hidden_dim,  # 保存第一个隐藏层大小。
        "hidden_dim2": hidden_dim2,  # 保存第二个隐藏层大小。
        "weight_decay": weight_decay,  # 保存 L2 正则化强度。
        "activation": activation,  # 保存激活函数。
        "batch_size": args.batch_size,  # 保存 batch size。
        "epochs": args.epochs,  # 保存训练 epoch 数。
        "decay_rate": args.decay_rate,  # 保存学习率衰减倍率。
        "decay_every": args.decay_every,  # 保存学习率衰减间隔。
        "best_val_acc": best_val_acc,  # 保存最佳验证集准确率。
        "best_val_loss": best_val_loss,  # 保存最佳验证集 loss。
        "best_train_acc": best_train_acc,  # 保存最佳 epoch 对应训练准确率。
        "best_train_loss": best_train_loss,  # 保存最佳 epoch 对应训练 loss。
        "best_epoch": best_epoch,  # 保存最佳 epoch。
        "checkpoint_path": str(checkpoint_path),  # 保存当前 run 的 best model 路径。
        "train_log_path": str(train_log_path),  # 保存当前 run 的训练日志路径。
        "elapsed_seconds": elapsed_seconds,  # 保存当前 run 耗时。
    }  # 当前 run 汇总结果字典创建完毕。
    print(f"Run {run_id:03d} 完成 | best_val_acc={best_val_acc:.4f} | best_epoch={best_epoch} | elapsed={elapsed_seconds:.1f}s")  # 打印当前 run 完成信息。
    return result  # 返回当前 run 的汇总结果。


def make_failed_result(run_id: int, config: Dict[str, object], args: argparse.Namespace, error: Exception) -> Dict[str, object]:  # 定义函数：当某组超参数实验失败时，构造失败记录。
    run_name = make_run_name(run_id, config)  # 根据 run_id 和 config 构造 run 名称。
    result = {  # 创建失败结果字典。
        "run_id": run_name,  # 保存 run 名称。
        "status": "failed",  # 标记当前 run 失败。
        "error_message": repr(error),  # 保存异常信息。
        "lr": float(config["lr"]),  # 保存学习率。
        "hidden_dim": int(config["hidden_dim"]),  # 保存隐藏层大小。
        "hidden_dim2": int(config["hidden_dim"]),  # 保存第二隐藏层大小。
        "weight_decay": float(config["weight_decay"]),  # 保存 L2 正则化强度。
        "activation": str(config["activation"]),  # 保存激活函数。
        "batch_size": args.batch_size,  # 保存 batch size。
        "epochs": args.epochs,  # 保存 epoch 数。
        "decay_rate": args.decay_rate,  # 保存学习率衰减倍率。
        "decay_every": args.decay_every,  # 保存学习率衰减间隔。
        "best_val_acc": "",  # 失败时没有最佳验证集准确率。
        "best_val_loss": "",  # 失败时没有最佳验证集 loss。
        "best_train_acc": "",  # 失败时没有最佳训练准确率。
        "best_train_loss": "",  # 失败时没有最佳训练 loss。
        "best_epoch": "",  # 失败时没有最佳 epoch。
        "checkpoint_path": "",  # 失败时没有模型权重路径。
        "train_log_path": "",  # 失败时没有训练日志路径。
        "elapsed_seconds": "",  # 失败时不记录耗时。
    }  # 失败结果字典创建完毕。
    return result  # 返回失败结果字典。


def save_best_summary(all_results: List[Dict[str, object]], search_dir: Path) -> Optional[Dict[str, object]]:  # 定义函数：根据所有 run 结果保存整体最佳组合 summary。
    successful_results = [result for result in all_results if result.get("status") == "ok"]  # 筛选所有成功完成的实验结果。
    if len(successful_results) == 0:  # 检查是否没有任何成功实验。
        return None  # 如果没有成功实验，就返回 None。
    best_result = max(successful_results, key=lambda result: float(result["best_val_acc"]))  # 按 best_val_acc 选择整体最佳结果。
    summary_path = search_dir / "best_hyperparams.json"  # 设置整体最佳超参数 summary 保存路径。
    best_overall_model_path = search_dir / "best_overall_model.npz"  # 设置整体最佳模型复制后的保存路径。
    checkpoint_path = Path(str(best_result["checkpoint_path"]))  # 读取整体最佳 run 的 checkpoint 路径。
    if checkpoint_path.exists():  # 检查整体最佳 checkpoint 是否真实存在。
        shutil.copy2(checkpoint_path, best_overall_model_path)  # 把整体最佳 checkpoint 复制成固定文件名。
        best_result["best_overall_model_path"] = str(best_overall_model_path)  # 把复制后的整体最佳模型路径写入结果字典。
    with summary_path.open("w", encoding="utf-8") as file:  # 以写入模式打开 summary JSON 文件。
        json.dump(best_result, file, ensure_ascii=False, indent=2)  # 把整体最佳结果保存成易读 JSON。
    return best_result  # 返回整体最佳结果字典。


def main() -> None:  # 定义主函数：组织完整超参数搜索流程。
    args = parse_args()  # 读取命令行参数。
    if args.batch_size <= 0:  # 检查 batch size 是否为正整数。
        raise ValueError(f"batch_size 必须为正整数，但当前 batch_size={args.batch_size}")  # 如果 batch size 不合法，就主动报错。
    if args.epochs <= 0:  # 检查 epoch 数是否为正整数。
        raise ValueError(f"epochs 必须为正整数，但当前 epochs={args.epochs}")  # 如果 epoch 数不合法，就主动报错。
    if args.decay_every <= 0:  # 检查学习率衰减间隔是否为正整数。
        raise ValueError(f"decay_every 必须为正整数，但当前 decay_every={args.decay_every}")  # 如果衰减间隔不合法，就主动报错。
    project_dir = Path(__file__).resolve().parent  # 获取当前 hyperparam_search.py 所在项目根目录。
    data_dir = resolve_path(args.data_dir, project_dir)  # 解析数据集目录路径。
    train_csv = resolve_path(args.train_csv, project_dir)  # 解析训练集 CSV 路径。
    val_csv = resolve_path(args.val_csv, project_dir)  # 解析验证集 CSV 路径。
    normalization_json = resolve_path(args.normalization_json, project_dir)  # 解析 normalization.json 路径。
    search_dir = resolve_path(args.search_dir, project_dir)  # 解析超参数搜索输出目录。
    results_csv_path = resolve_path(args.results_csv, project_dir)  # 解析超参数搜索汇总 CSV 路径。
    check_training_files(data_dir, train_csv, val_csv, normalization_json)  # 检查训练和验证所需文件是否存在。
    search_dir.mkdir(parents=True, exist_ok=True)  # 创建超参数搜索输出目录。
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)  # 创建汇总 CSV 所在目录。
    learning_rates = parse_float_list(args.learning_rates)  # 解析学习率候选列表。
    hidden_dims = parse_int_list(args.hidden_dims)  # 解析隐藏层大小候选列表。
    weight_decays = parse_float_list(args.weight_decays)  # 解析 L2 正则化强度候选列表。
    activations = parse_str_list(args.activations)  # 解析激活函数候选列表。
    validate_activation_names(activations)  # 检查激活函数候选列表是否合法。
    full_grid = build_search_grid(learning_rates, hidden_dims, weight_decays, activations)  # 构造完整超参数组合网格。
    selected_configs = select_search_configs(full_grid, args.search_mode, args.max_trials, args.seed)  # 根据搜索模式选择实际要跑的组合。
    mean_rgb, std_rgb = load_normalization_params(normalization_json)  # 读取训练集 RGB 均值和标准差。
    normalization_params = load_json_dict(normalization_json)  # 读取完整 normalization.json，用于类别数和保存模型元信息。
    train_samples = read_split_csv(train_csv, data_dir)  # 读取训练集样本列表。
    val_samples = read_split_csv(val_csv, data_dir)  # 读取验证集样本列表。
    if len(train_samples) == 0:  # 检查训练集是否为空。
        raise ValueError("训练集为空，无法进行超参数搜索。")  # 如果训练集为空，就主动报错。
    if len(val_samples) == 0:  # 检查验证集是否为空。
        raise ValueError("验证集为空，无法进行超参数搜索。")  # 如果验证集为空，就主动报错。
    write_search_header(results_csv_path)  # 创建超参数搜索汇总 CSV，并写入表头。
    all_results: List[Dict[str, object]] = []  # 创建空列表，用来保存所有 run 的结果字典。
    print("超参数搜索模式：", args.search_mode)  # 打印搜索模式。
    print("完整组合数量：", len(full_grid))  # 打印完整网格组合数量。
    print("实际运行组合数量：", len(selected_configs))  # 打印实际运行组合数量。
    print("训练集样本数：", len(train_samples))  # 打印训练集样本数。
    print("验证集样本数：", len(val_samples))  # 打印验证集样本数。
    print("搜索结果 CSV：", results_csv_path)  # 打印搜索结果 CSV 保存路径。
    for run_index, config in enumerate(selected_configs, start=1):  # 逐组运行被选中的超参数组合。
        try:  # 开始尝试运行当前超参数组合。
            result = run_single_config(run_index, config, args, data_dir, train_samples, val_samples, mean_rgb, std_rgb, normalization_params, search_dir)  # 训练并评估当前超参数组合。
        except Exception as error:  # 捕获当前组合训练过程中的异常。
            print(f"Run {run_index:03d} 失败：{repr(error)}")  # 在终端打印当前 run 失败信息。
            result = make_failed_result(run_index, config, args, error)  # 构造失败结果记录。
        append_search_result(results_csv_path, result)  # 把当前 run 结果追加写入汇总 CSV。
        all_results.append(result)  # 把当前 run 结果加入内存列表。
    best_result = save_best_summary(all_results, search_dir)  # 保存整体最佳超参数组合 summary。
    print("=" * 100)  # 打印分隔线。
    print("超参数搜索结束。")  # 打印搜索结束提示。
    print("汇总结果已保存到：", results_csv_path)  # 打印汇总 CSV 路径。
    if best_result is not None:  # 判断是否存在成功的整体最佳结果。
        print("整体最佳 run：", best_result["run_id"])  # 打印整体最佳 run 名称。
        print("整体最佳验证集准确率：", best_result["best_val_acc"])  # 打印整体最佳验证集准确率。
        print("整体最佳超参数：", {"lr": best_result["lr"], "hidden_dim": best_result["hidden_dim"], "weight_decay": best_result["weight_decay"], "activation": best_result["activation"]})  # 打印整体最佳超参数组合。
        print("整体最佳模型路径：", best_result.get("best_overall_model_path", best_result["checkpoint_path"]))  # 打印整体最佳模型路径。
    else:  # 如果所有 run 都失败。
        print("没有任何成功完成的 run，请检查错误信息和超参数设置。")  # 打印失败提示。


if __name__ == "__main__":  # 只有直接运行 hyperparam_search.py 时，才执行主流程。
    main()  # 调用主函数，开始超参数搜索。
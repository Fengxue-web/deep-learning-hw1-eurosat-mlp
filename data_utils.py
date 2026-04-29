from pathlib import Path  # 导入 Path，用来更安全、更方便地处理 Windows / Linux / macOS 上的文件路径。
from typing import Dict, List, Sequence, Tuple, Union  # 导入类型标注工具，方便说明函数输入输出的数据类型。
import numpy as np  # 导入 NumPy，后续用它把图片转换成数组，并为 MLP 准备输入向量。
from PIL import Image  # 导入 Pillow 中的 Image 类，用来读取 JPG 图片并转换为 RGB 格式。
import csv  # 导入 Python 标准库 csv，用来把训练集、验证集、测试集的划分结果保存成 CSV 文件。
import json  # 导入 Python 标准库 json，用来把类别映射、均值、标准差等配置信息保存成 JSON 文件。

IMAGE_SIZE: Tuple[int, int] = (64, 64)  # 设置统一的图片尺寸；Pillow 的 resize 使用的是“宽, 高”，这里宽和高都是 64。
ALLOWED_IMAGE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png")  # 设置允许读取的图片后缀；虽然数据是 JPG，但这里顺手兼容 jpeg/png。
BILINEAR_RESAMPLE = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR  # 设置 resize 插值方法，并兼容新旧版本 Pillow。


def scan_dataset(root_dir: Union[str, Path], allowed_suffixes: Sequence[str] = ALLOWED_IMAGE_SUFFIXES) -> Tuple[List[Tuple[str, int, str]], Dict[str, int], Dict[int, str]]:  # 定义扫描数据集目录的函数，返回样本列表、类别到编号的字典、编号到类别的字典。
    root_path = Path(root_dir)  # 把传入的字符串路径转换成 Path 对象，后续可以用 Path 的方法处理路径。
    if not root_path.exists():  # 检查 EuroSAT_RGB 根目录是否真的存在，避免路径写错后程序静默失败。
        raise FileNotFoundError(f"数据集根目录不存在：{root_path}")  # 如果根目录不存在，就主动报错并显示错误路径。
    if not root_path.is_dir():  # 检查传入路径是否是文件夹，而不是某个具体文件。
        raise NotADirectoryError(f"传入的 root_dir 不是文件夹：{root_path}")  # 如果不是文件夹，就主动报错。
    class_dirs = [path for path in root_path.iterdir() if path.is_dir()]  # 扫描 EuroSAT_RGB 下面的所有子文件夹，每个子文件夹对应一个类别。
    class_dirs = sorted(class_dirs, key=lambda path: path.name)  # 按类别文件夹名称排序，保证每次运行时类别编号顺序都一致。
    if len(class_dirs) == 0:  # 检查是否真的找到了类别文件夹。
        raise ValueError(f"在数据集根目录下没有找到任何类别文件夹：{root_path}")  # 如果没有找到类别文件夹，就说明数据集目录层级可能放错了。
    class_to_idx = {class_dir.name: idx for idx, class_dir in enumerate(class_dirs)}  # 为每个类别名称分配一个整数标签，例如 AnnualCrop -> 0。
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}  # 建立反向映射，例如 0 -> AnnualCrop，方便之后解释预测结果。
    samples: List[Tuple[str, int, str]] = []  # 创建空列表，用来保存所有样本，每个样本格式为“图片路径、整数标签、类别名称”。
    for class_dir in class_dirs:  # 逐个遍历类别文件夹，例如 AnnualCrop、Forest、River 等。
        class_name = class_dir.name  # 取出当前类别文件夹的名称，作为类别名。
        label_id = class_to_idx[class_name]  # 根据类别名查出对应的整数标签。
        image_paths: List[Path] = []  # 创建空列表，用来保存当前类别文件夹下所有图片路径。
        for suffix in allowed_suffixes:  # 遍历允许的图片后缀，例如 .jpg、.jpeg、.png。
            image_paths.extend(class_dir.glob(f"*{suffix}"))  # 查找当前类别文件夹下小写后缀的图片文件，例如 .jpg。
            image_paths.extend(class_dir.glob(f"*{suffix.upper()}"))  # 查找当前类别文件夹下大写后缀的图片文件，例如 .JPG。
        image_paths = sorted(set(image_paths), key=lambda path: path.name)  # 去重并按文件名排序，保证每次扫描得到的图片顺序一致。
        if len(image_paths) == 0:  # 检查当前类别文件夹下面是否真的有图片。
            print(f"警告：类别文件夹 {class_dir} 下没有找到图片文件。")  # 如果某个类别没有图片，打印警告，但不中断整个扫描。
        for image_path in image_paths:  # 逐个遍历当前类别下的所有图片路径。
            samples.append((str(image_path), label_id, class_name))  # 把“图片路径、标签编号、类别名称”作为一个样本加入总样本列表。
    if len(samples) == 0:  # 检查整个数据集中是否至少找到了一张图片。
        raise ValueError(f"没有扫描到任何图片，请检查数据集路径和图片后缀：{root_path}")  # 如果一张图片都没找到，就主动报错。
    return samples, class_to_idx, idx_to_class  # 返回样本列表、类别名到标签编号的映射、标签编号到类别名的映射。


def load_image(image_path: Union[str, Path], image_size: Tuple[int, int] = IMAGE_SIZE, normalize: bool = True) -> np.ndarray:  # 定义读取单张图片的函数，输出形状为 H × W × 3 的 NumPy 数组。
    path = Path(image_path)  # 把输入的图片路径转换成 Path 对象，方便统一处理。
    if not path.exists():  # 检查图片文件是否真实存在。
        raise FileNotFoundError(f"图片文件不存在：{path}")  # 如果图片路径不存在，就主动报错并显示错误路径。
    if not path.is_file():  # 检查路径是否是文件，而不是文件夹。
        raise ValueError(f"传入的 image_path 不是文件：{path}")  # 如果不是文件，就主动报错。
    with Image.open(path) as img:  # 用 Pillow 打开图片，并用 with 保证读取结束后自动关闭文件。
        img = img.convert("RGB")  # 把图片强制转换为 RGB 三通道，避免灰度图、调色板图等模式导致通道数不一致。
        img = img.resize(image_size, resample=BILINEAR_RESAMPLE)  # 把图片统一 resize 到 64×64，保证后续 MLP 输入维度固定。
        img_array = np.asarray(img, dtype=np.float32)  # 把 Pillow 图片转换成 NumPy 数组，并把像素类型转换为 float32。
    if normalize:  # 判断是否需要把像素从 0~255 缩放到 0~1。
        img_array = img_array / 255.0  # 把像素值除以 255，使输入数值范围变成 0~1，训练会更稳定。
    return img_array  # 返回处理后的图片数组，形状通常是 (64, 64, 3)。


def image_to_mlp_input(img_array: np.ndarray) -> np.ndarray:  # 定义把图片数组展平成 MLP 输入向量的函数。
    if img_array.ndim != 3:  # 检查图片数组是否是三维数组，即 H × W × C。
        raise ValueError(f"图片数组应该是三维的 H×W×C，但当前 shape 是：{img_array.shape}")  # 如果不是三维，就报错说明当前形状。
    if img_array.shape[2] != 3:  # 检查第三个维度是否是 RGB 三通道。
        raise ValueError(f"图片数组最后一维应该是 3 个 RGB 通道，但当前 shape 是：{img_array.shape}")  # 如果不是三通道，就报错说明当前形状。
    x = img_array.reshape(-1).astype(np.float32, copy=False)  # 把 H×W×3 的图片展平成一维向量，例如 64×64×3 会变成 12288 维。
    return x  # 返回 MLP 可以直接接收的一维输入向量。


def encode_label(class_name: str, class_to_idx: Dict[str, int]) -> int:  # 定义把类别名称转换成整数标签的函数。
    if class_name not in class_to_idx:  # 检查类别名称是否在类别映射表中。
        raise KeyError(f"类别 {class_name} 不在 class_to_idx 中，可选类别为：{list(class_to_idx.keys())}")  # 如果类别名不存在，就报错并列出可选类别。
    label_id = class_to_idx[class_name]  # 从类别映射表中取出该类别对应的整数标签。
    return label_id  # 返回整数标签，例如 AnnualCrop 可能对应 0。


def load_sample(sample: Tuple[str, int, str], image_size: Tuple[int, int] = IMAGE_SIZE) -> Tuple[np.ndarray, int]:  # 定义读取一个样本的函数，把“路径样本”变成“模型输入向量和标签”。
    image_path, label_id, class_name = sample  # 从样本元组中拆出图片路径、整数标签、类别名称。
    img_array = load_image(image_path, image_size=image_size, normalize=True)  # 读取图片，转换 RGB，resize，并把像素缩放到 0~1。
    x = image_to_mlp_input(img_array)  # 把 64×64×3 的图片数组展平成 12288 维向量。
    y = int(label_id)  # 把标签编号明确转换成 Python 整数，方便后续交叉熵损失函数使用。
    return x, y  # 返回一个 MLP 输入向量 x 和一个整数标签 y。


Sample = Tuple[str, int, str]  # 定义一个类型别名；一个样本由“图片路径、整数标签、类别名称”三个元素组成。


def group_samples_by_class(samples: Sequence[Sample]) -> Dict[int, List[Sample]]:  # 定义函数：按照整数标签把样本分组，方便后面做分层划分。
    grouped_samples: Dict[int, List[Sample]] = {}  # 创建一个空字典；键是类别标签，值是该类别下的样本列表。
    for sample in samples:  # 遍历所有样本；每个 sample 的格式是“图片路径、整数标签、类别名称”。
        image_path, label_id, class_name = sample  # 把当前样本拆成图片路径、整数标签、类别名称三个变量。
        if label_id not in grouped_samples:  # 判断当前标签是否还没有出现在分组字典中。
            grouped_samples[label_id] = []  # 如果该标签还没有对应的列表，就先创建一个空列表。
        grouped_samples[label_id].append(sample)  # 把当前样本加入它所属类别对应的样本列表中。
    return grouped_samples  # 返回分组后的结果，例如 0 类一组、1 类一组、一直到 9 类一组。


def shuffle_samples(samples: Sequence[Sample], rng: np.random.Generator) -> List[Sample]:  # 定义函数：使用给定随机数生成器打乱样本顺序。
    indices = rng.permutation(len(samples))  # 生成一个 0 到 len(samples)-1 的随机排列，用来表示打乱后的索引顺序。
    shuffled_samples = [samples[int(index)] for index in indices]  # 按照随机排列后的索引顺序重新取样本，从而得到打乱后的样本列表。
    return shuffled_samples  # 返回打乱顺序后的样本列表。


def stratified_split(samples: Sequence[Sample], train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42) -> Tuple[List[Sample], List[Sample], List[Sample]]:  # 定义分层划分函数；默认按 70%、15%、15% 划分训练、验证、测试集。
    ratio_sum = train_ratio + val_ratio + test_ratio  # 计算三个比例的总和，正常情况下应该等于 1。
    if not np.isclose(ratio_sum, 1.0):  # 判断三个比例之和是否足够接近 1，避免用户误写比例。
        raise ValueError(f"train_ratio + val_ratio + test_ratio 必须等于 1，但当前等于 {ratio_sum}")  # 如果比例之和不为 1，就主动报错。
    rng = np.random.default_rng(seed)  # 创建 NumPy 随机数生成器；固定 seed 可以保证每次划分结果完全一致。
    grouped_samples = group_samples_by_class(samples)  # 先按照类别标签分组，确保每个类别都按同样比例划分。
    train_samples: List[Sample] = []  # 创建训练集样本列表，后面会逐类加入训练样本。
    val_samples: List[Sample] = []  # 创建验证集样本列表，后面会逐类加入验证样本。
    test_samples: List[Sample] = []  # 创建测试集样本列表，后面会逐类加入测试样本。
    for label_id in sorted(grouped_samples.keys()):  # 按类别标签从小到大遍历，保证划分过程稳定、可复现。
        class_samples = list(grouped_samples[label_id])  # 取出当前类别的全部样本，并复制成一个新的列表。
        class_samples = shuffle_samples(class_samples, rng)  # 只在当前类别内部打乱样本，保证每个类别内部随机划分。
        n_total = len(class_samples)  # 统计当前类别一共有多少张图片。
        n_train = int(n_total * train_ratio)  # 计算当前类别分到训练集的样本数量。
        n_val = int(n_total * val_ratio)  # 计算当前类别分到验证集的样本数量。
        train_part = class_samples[:n_train]  # 从当前类别打乱后的样本中取前 n_train 个作为训练集部分。
        val_part = class_samples[n_train:n_train + n_val]  # 从训练集之后继续取 n_val 个作为验证集部分。
        test_part = class_samples[n_train + n_val:]  # 剩下的样本全部作为测试集部分，避免因为取整造成样本丢失。
        train_samples.extend(train_part)  # 把当前类别的训练集部分加入总训练集。
        val_samples.extend(val_part)  # 把当前类别的验证集部分加入总验证集。
        test_samples.extend(test_part)  # 把当前类别的测试集部分加入总测试集。
    train_samples = shuffle_samples(train_samples, rng)  # 对所有类别汇总后的训练集再次整体打乱，避免训练时类别按顺序集中出现。
    val_samples = shuffle_samples(val_samples, rng)  # 对所有类别汇总后的验证集再次整体打乱，方便后续评估时顺序更随机。
    test_samples = shuffle_samples(test_samples, rng)  # 对所有类别汇总后的测试集再次整体打乱，方便后续测试时顺序更随机。
    return train_samples, val_samples, test_samples  # 返回训练集、验证集、测试集三个样本列表。


def sample_to_relative_path(image_path: Union[str, Path], root_dir: Union[str, Path]) -> str:  # 定义函数：把绝对图片路径转换成相对于 EuroSAT_RGB 根目录的相对路径。
    path = Path(image_path)  # 把图片路径转换成 Path 对象，方便做路径运算。
    root_path = Path(root_dir)  # 把数据集根目录转换成 Path 对象，方便做路径运算。
    relative_path = path.resolve().relative_to(root_path.resolve())  # 计算图片路径相对于数据集根目录的相对路径。
    relative_path_string = relative_path.as_posix()  # 把路径统一转换成正斜杠格式，避免 Windows 反斜杠在 CSV 中不够通用。
    return relative_path_string  # 返回相对路径字符串，例如 AnnualCrop/AnnualCrop_1.jpg。


def save_split_csv(samples: Sequence[Sample], csv_path: Union[str, Path], root_dir: Union[str, Path]) -> None:  # 定义函数：把某一个划分结果保存成 CSV 文件。
    csv_path = Path(csv_path)  # 把输出 CSV 路径转换成 Path 对象。
    csv_path.parent.mkdir(parents=True, exist_ok=True)  # 确保 CSV 文件所在的文件夹存在；如果不存在就自动创建。
    with csv_path.open("w", newline="", encoding="utf-8") as file:  # 以写入模式打开 CSV 文件，并使用 UTF-8 编码保存中文或特殊字符。
        writer = csv.writer(file)  # 创建 CSV 写入器，用来逐行写入数据。
        writer.writerow(["relative_path", "label", "class_name"])  # 写入表头；相对路径、整数标签、类别名称。
        for sample in samples:  # 遍历当前划分中的每一个样本。
            image_path, label_id, class_name = sample  # 把样本拆成图片路径、整数标签、类别名称。
            relative_path = sample_to_relative_path(image_path, root_dir)  # 把图片绝对路径转换成相对路径，方便项目换电脑后仍然可用。
            writer.writerow([relative_path, int(label_id), class_name])  # 把当前样本的一行信息写入 CSV 文件。


def save_all_splits(train_samples: Sequence[Sample], val_samples: Sequence[Sample], test_samples: Sequence[Sample], output_dir: Union[str, Path], root_dir: Union[str, Path]) -> Dict[str, str]:  # 定义函数：同时保存 train/val/test 三个 CSV 文件。
    output_dir = Path(output_dir)  # 把输出文件夹路径转换成 Path 对象。
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出文件夹；如果已经存在，就不会报错。
    train_csv_path = output_dir / "train.csv"  # 设置训练集 CSV 的保存路径。
    val_csv_path = output_dir / "val.csv"  # 设置验证集 CSV 的保存路径。
    test_csv_path = output_dir / "test.csv"  # 设置测试集 CSV 的保存路径。
    save_split_csv(train_samples, train_csv_path, root_dir)  # 保存训练集划分结果到 train.csv。
    save_split_csv(val_samples, val_csv_path, root_dir)  # 保存验证集划分结果到 val.csv。
    save_split_csv(test_samples, test_csv_path, root_dir)  # 保存测试集划分结果到 test.csv。
    split_paths = {"train": str(train_csv_path), "val": str(val_csv_path), "test": str(test_csv_path)}  # 把三个 CSV 路径整理成字典，方便打印或记录。
    return split_paths  # 返回三个划分文件的路径字典。


def read_split_csv(csv_path: Union[str, Path], root_dir: Union[str, Path]) -> List[Sample]:  # 定义函数：从保存好的 CSV 文件中重新读取样本列表，后续训练脚本会用到。
    csv_path = Path(csv_path)  # 把 CSV 路径转换成 Path 对象。
    root_path = Path(root_dir)  # 把数据集根目录转换成 Path 对象。
    samples: List[Sample] = []  # 创建空列表，用来存放从 CSV 中读回来的样本。
    with csv_path.open("r", newline="", encoding="utf-8") as file:  # 以读取模式打开 CSV 文件。
        reader = csv.DictReader(file)  # 创建字典形式的 CSV 读取器，可以按列名读取每一行。
        for row in reader:  # 遍历 CSV 文件中的每一行。
            image_path = root_path / row["relative_path"]  # 把相对路径拼接回完整图片路径。
            label_id = int(row["label"])  # 把 CSV 中读出的标签字符串转换成整数。
            class_name = row["class_name"]  # 读取当前样本的类别名称。
            samples.append((str(image_path), label_id, class_name))  # 把恢复出的样本加入样本列表。
    return samples  # 返回从 CSV 文件中恢复出来的样本列表。


def count_samples_by_class(samples: Sequence[Sample]) -> Dict[str, int]:  # 定义函数：统计某个样本列表中每个类别有多少张图片。
    counts: Dict[str, int] = {}  # 创建空字典；键是类别名称，值是该类别下的样本数量。
    for sample in samples:  # 遍历样本列表中的每一个样本。
        image_path, label_id, class_name = sample  # 把样本拆成图片路径、整数标签、类别名称。
        counts[class_name] = counts.get(class_name, 0) + 1  # 对当前类别计数加 1；如果之前没出现过，就从 0 开始。
    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[0]))  # 按类别名称排序，让打印结果更稳定、更易读。
    return sorted_counts  # 返回类别计数字典。


def compute_rgb_mean_std(train_samples: Sequence[Sample], image_size: Tuple[int, int] = IMAGE_SIZE) -> Dict[str, object]:  # 定义函数：只基于训练集计算 RGB 三个通道的均值和标准差。
    if len(train_samples) == 0:  # 检查训练集是否为空。
        raise ValueError("训练集为空，无法计算归一化参数。")  # 如果训练集为空，就主动报错。
    pixel_sum = np.zeros(3, dtype=np.float64)  # 创建长度为 3 的数组，用来累加 R、G、B 三个通道的像素和。
    pixel_square_sum = np.zeros(3, dtype=np.float64)  # 创建长度为 3 的数组，用来累加 R、G、B 三个通道的像素平方和。
    pixel_count = 0  # 创建计数器，用来记录训练集中总共统计了多少个像素位置。
    for sample in train_samples:  # 遍历训练集中的每一个样本。
        image_path, label_id, class_name = sample  # 把当前样本拆成图片路径、整数标签、类别名称。
        img_array = load_image(image_path, image_size=image_size, normalize=True)  # 读取图片，并先把像素从 0~255 缩放到 0~1。
        img_array = img_array.astype(np.float64, copy=False)  # 把图片数组转换成 float64，减少大量累加时的数值误差。
        height, width, channels = img_array.shape  # 读取图片数组的高度、宽度、通道数。
        if channels != 3:  # 检查图片是否是 RGB 三通道。
            raise ValueError(f"图片不是 RGB 三通道：{image_path}, shape={img_array.shape}")  # 如果不是三通道，就主动报错。
        pixel_sum += img_array.sum(axis=(0, 1))  # 沿高度和宽度两个维度求和，得到当前图片 R、G、B 三个通道的像素和。
        pixel_square_sum += (img_array * img_array).sum(axis=(0, 1))  # 沿高度和宽度求平方和，得到当前图片 R、G、B 三个通道的像素平方和。
        pixel_count += height * width  # 当前图片贡献 height × width 个像素位置，每个位置都有 RGB 三个通道。
    mean_rgb = pixel_sum / pixel_count  # 用通道像素总和除以像素位置总数，得到 R、G、B 三个通道的均值。
    variance_rgb = pixel_square_sum / pixel_count - mean_rgb * mean_rgb  # 根据 E[X^2] - E[X]^2 计算 R、G、B 三个通道的方差。
    variance_rgb = np.maximum(variance_rgb, 1e-12)  # 防止因为浮点误差导致极小的负方差，从而影响开平方。
    std_rgb = np.sqrt(variance_rgb)  # 对方差开平方，得到 R、G、B 三个通道的标准差。
    params = {"image_size": list(image_size), "pixel_scale": 255.0, "mean_rgb": mean_rgb.tolist(), "std_rgb": std_rgb.tolist(), "num_train_images": len(train_samples), "num_pixels_per_channel": int(pixel_count)}  # 把统计结果整理成可保存为 JSON 的字典。
    return params  # 返回归一化 / 标准化参数字典。


def save_json_dict(data: Dict[str, object], json_path: Union[str, Path]) -> None:  # 定义函数：把字典保存成 JSON 文件。
    json_path = Path(json_path)  # 把 JSON 输出路径转换成 Path 对象。
    json_path.parent.mkdir(parents=True, exist_ok=True)  # 确保 JSON 文件所在的目录存在；如果不存在就自动创建。
    with json_path.open("w", encoding="utf-8") as file:  # 以写入模式打开 JSON 文件，并使用 UTF-8 编码。
        json.dump(data, file, ensure_ascii=False, indent=2)  # 把字典写入 JSON 文件；ensure_ascii=False 可以保留中文，indent=2 让文件更易读。


def print_split_summary(train_samples: Sequence[Sample], val_samples: Sequence[Sample], test_samples: Sequence[Sample]) -> None:  # 定义函数：打印训练集、验证集、测试集的数量摘要。
    print("训练集总样本数：", len(train_samples))  # 打印训练集样本数量。
    print("验证集总样本数：", len(val_samples))  # 打印验证集样本数量。
    print("测试集总样本数：", len(test_samples))  # 打印测试集样本数量。
    print("训练集各类别数量：", count_samples_by_class(train_samples))  # 打印训练集中每个类别的样本数量。
    print("验证集各类别数量：", count_samples_by_class(val_samples))  # 打印验证集中每个类别的样本数量。
    print("测试集各类别数量：", count_samples_by_class(test_samples))  # 打印测试集中每个类别的样本数量。


if __name__ == "__main__":  # 只有直接运行本文件时，下面这段测试代码才会执行；被其他文件 import 时不会执行。
    root_dir = Path(__file__).resolve().parent / "data" / "EuroSAT_RGB"  # 假设 data_utils.py 放在项目根目录，数据集放在 data/EuroSAT_RGB。
    samples, class_to_idx, idx_to_class = scan_dataset(root_dir)  # 扫描类别文件夹和图片文件，生成样本列表与标签映射。
    print("类别到整数标签的映射 class_to_idx：", class_to_idx)  # 打印类别名称到整数标签的映射，检查标签编码是否合理。
    print("整数标签到类别的映射 idx_to_class：", idx_to_class)  # 打印整数标签到类别名称的映射，方便之后解释预测结果。
    print("扫描到的总图片数量：", len(samples))  # 打印总样本数，EuroSAT_RGB 完整数据集通常应为 27000 张左右。
    first_image_path, first_label_id, first_class_name = samples[0]  # 取出扫描到的第一张图片，用来做一次读取和展平测试。
    with Image.open(first_image_path) as img:  # 直接打开第一张原始图片，用来查看它未经 resize 前的尺寸和模式。
        print("第一张原始图片路径：", first_image_path)  # 打印第一张图片的文件路径，方便定位。
        print("第一张原始图片尺寸 img.size：", img.size)  # 打印 Pillow 看到的原始尺寸，格式是“宽, 高”。
        print("第一张原始图片模式 img.mode：", img.mode)  # 打印原始图片模式，例如 RGB。
    img_array = load_image(first_image_path, image_size=IMAGE_SIZE, normalize=True)  # 按正式训练流程读取第一张图片。
    x = image_to_mlp_input(img_array)  # 把第一张图片展平成 MLP 输入向量。
    y = encode_label(first_class_name, class_to_idx)  # 用类别名称重新编码一次标签，检查 encode_label 函数。
    print("第一张图片类别名称：", first_class_name)  # 打印第一张图片所属类别。
    print("第一张图片扫描得到的标签：", first_label_id)  # 打印 scan_dataset 阶段记录的标签。
    print("第一张图片重新编码的标签：", y)  # 打印 encode_label 重新得到的标签，应与 first_label_id 一致。
    print("处理后图片数组 shape：", img_array.shape)  # 打印处理后的图片数组形状，应为 (64, 64, 3)。
    print("展平后的 MLP 输入 shape：", x.shape)  # 打印展平后的输入向量形状，应为 (12288,)。
    print("展平后前 10 个像素值：", x[:10])  # 打印前 10 个归一化像素值，用来确认数值已经在 0~1 之间。
    project_dir = Path(__file__).resolve().parent  # 获取当前 data_utils.py 所在的项目根目录。
    root_dir = project_dir / "data" / "EuroSAT_RGB"  # 设置 EuroSAT_RGB 数据集根目录，与当前项目结构保持一致。
    output_dir = project_dir / "outputs"  # 设置所有输出文件的总目录，例如划分结果和归一化参数都会放在这里。
    splits_dir = output_dir / "splits"  # 设置训练集、验证集、测试集 CSV 文件的保存目录。
    normalization_json_path = output_dir / "normalization.json"  # 设置归一化 / 标准化参数 JSON 文件的保存路径。
    train_samples, val_samples, test_samples = stratified_split(samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)  # 按类别分层划分训练集、验证集、测试集。
    split_paths = save_all_splits(train_samples, val_samples, test_samples, splits_dir, root_dir)  # 把三个划分结果分别保存成 train.csv、val.csv、test.csv。
    normalization_params = compute_rgb_mean_std(train_samples, image_size=IMAGE_SIZE)  # 只使用训练集计算 RGB 通道均值和标准差，避免验证集和测试集信息泄露。
    normalization_params["class_to_idx"] = class_to_idx  # 把类别名称到整数标签的映射也保存进 normalization.json，方便后续训练和测试统一使用。
    normalization_params["idx_to_class"] = {str(idx): class_name for idx, class_name in idx_to_class.items()}  # 把整数标签到类别名称的映射也保存进去；JSON 的键建议使用字符串。
    normalization_params["split_ratio"] = {"train": 0.70, "val": 0.15, "test": 0.15}  # 保存本次数据集划分比例，方便实验报告和 README 说明。
    normalization_params["seed"] = 42  # 保存随机种子，方便以后完全复现同一次数据集划分。
    normalization_params["note"] = "mean_rgb 和 std_rgb 是在训练集上、对已经除以 255.0 的 RGB 图片计算得到的；验证集、测试集、预测图片都应使用同一组参数。"  # 保存文字说明，提醒后续不能重新用验证集或测试集计算标准化参数。
    save_json_dict(normalization_params, normalization_json_path)  # 把归一化 / 标准化参数保存成 JSON 文件。
    print("数据集根目录：", root_dir)  # 打印数据集根目录，方便检查路径是否正确。
    print("扫描到的总样本数：", len(samples))  # 打印扫描到的总图片数量，完整 EuroSAT_RGB 通常应为 27000。
    print("类别到整数标签的映射：", class_to_idx)  # 打印类别到标签的映射，方便确认类别编码是否稳定。
    print_split_summary(train_samples, val_samples, test_samples)  # 打印训练集、验证集、测试集的总数和类别分布。
    print("划分结果保存路径：", split_paths)  # 打印 train.csv、val.csv、test.csv 的保存路径。
    print("归一化参数保存路径：", normalization_json_path)  # 打印 normalization.json 的保存路径。
    print("训练集 RGB 均值 mean_rgb：", normalization_params["mean_rgb"])  # 打印训练集 RGB 均值，后续标准化时会用到。
    print("训练集 RGB 标准差 std_rgb：", normalization_params["std_rgb"])  # 打印训练集 RGB 标准差，后续标准化时会用到。
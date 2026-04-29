# deep-learning-hw1-eurosat-mlp
任务描述：手工搭建三层神经网络 (MLP) 分类器，在遥感图像数据集 EuroSAT(数据集文件见
EuroSAT_RGB 文件夹) 上进行训练，实现基于卫星图像的土地覆盖分类。EuroSAT
包含 10 个类别 (森林、河流、高速公路、住宅区等)。

环境依赖：

Python >= 3.10

numpy

Pillow

matplotlib

如何运行训练和测试脚本：

step1：运行data_utils.py

step2：运行hyperparam_search.py --search_mode grid --max_trials 0 --epochs 30

step3：运行evaluate.py

# deep-learning-hw1-eurosat-mlp
NumPy implementation of a three-layer MLP for EuroSAT land-cover classification

环境依赖：

Python >= 3.10; 

numpy; 

Pillow; 

matplotlib.

如何运行训练和测试脚本：

step1：运行data_utils.py; 

step2：运行hyperparam_search.py --search_mode grid --max_trials 0 --epochs 30; 

step3：运行evaluate.py.

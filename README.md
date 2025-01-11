
## Installation
代码运行所需环境 PyTorch 1.11.0 and CUDA 11.3. 请按照如下步骤安装,

1. 创建Conda环境
```shell
conda create --name unetr_pp python=3.8
conda activate unetr_pp
```
2. 安装PyTorch 和torchvision
```shell
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
3. 安装其他依赖环境
```shell
pip install -r requirements.txt
```
## Dataset
采用HIE私有数据集
## Training
运行脚本
```shell
bash training_scripts/run_training_tumor.sh
```
## Evaluation
运行脚本
```shell
bash evaluation_scripts/run_evaluation_tumor.sh
```

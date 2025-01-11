import os
import subprocess

# 设置路径
DATASET_PATH = "F:/UNETR++ code/unetr_plus_plus-main/DATASET_Tumor"

os.environ["PYTHONPATH"] = ".././"
os.environ["RESULTS_FOLDER"] = "F:/UNETR++ code/unetr_plus_plus-main/unetr_pp/evaluation/unetr_pp_tumor_checkpoint"
os.environ["unetr_pp_preprocessed"] = os.path.join(DATASET_PATH, "unetr_pp_raw/unetr_pp_raw_data/Task03_tumor")
os.environ["unetr_pp_raw_data_base"] = os.path.join(DATASET_PATH, "unetr_pp_raw")
print(os.path.join(DATASET_PATH, "unetr_pp_raw/unetr_pp_raw_data/Task03_tumor"))
print(os.path.join(DATASET_PATH, "unetr_pp_raw"))

# 提示信息
print("Note: For Tumor, it is recommended to train unetr_plus_plus first, and then use the provided checkpoint to evaluate. "
      "It might raise issues regarding the pickle files if you evaluated without training.")

# 运行第一个命令
predict_command = [
    "python", "F:/UNETR++ code/unetr_plus_plus-main/unetr_pp/inference/predict_simple.py",
    "-i", "F:/UNETR++ code/unetr_plus_plus-main/DATASET_Tumor/unetr_pp_raw/unetr_pp_raw_data/Task003_tumor/imagesTs",
    "-o", "F:/UNETR++ code/unetr_plus_plus-main/unetr_pp/evaluation/unetr_pp_tumor_checkpoint/inferTs",
    "-m", "3d_fullres",
    "-t", "3",
    "-f", "0",
    "-chk", "model_final_checkpoint",
    "-tr", "unetr_pp_trainer_tumor"
]

print("Running prediction command...")
subprocess.run(predict_command)

# 运行第二个命令
inference_command = [
    "python", "../unetr_pp/inference_tumor.py", "0"
]

print("Running inference command...")
subprocess.run(inference_command)

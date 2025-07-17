import os

# 文件路径
DATA_PATH = os.path.join("data", "US-pumpkins.csv")
OUTPUT_CSV_PATH = os.path.join("output", "output.csv")
OUTPUT_JSON_PATH = os.path.join("output", "output.json")

# 特征与目标列
FEATURE_COLUMNS = [
    "City Name", "Package", "Variety", "Origin", "Item Size", 
    "Color", "Date"  # 后续会处理为年/月
]
TARGET_COLUMN = "mean_price"  # 由Low Price和High Price计算

# 模型参数
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "random_state": 42
}

XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42
}

# 交叉验证配置
CV_FOLDS = 3
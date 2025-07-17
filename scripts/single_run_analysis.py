import os
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_processing import process_features
from model import get_lgbm_model, get_xgboost_model
from evaluate import evaluate_performance
from configuration import LGBM_PARAMS, XGBOOST_PARAMS

def run_single_analysis(
    data, 
    feature_columns, 
    model_name="LGBM", 
    output_path="output/single_run_results.csv",
):
    """
    单轮实验分析：输出训练集+测试集预测结果，格式与示例对齐
    参数：
        output_path: 主结果路径（训练/测试集文件会保存在同目录下）
    """
    # 1. 特征处理
    X, y, _ = process_features(data, feature_columns)
    print(f"[单轮分析] 特征处理完成 | 特征维度: {X.shape}")

    # 2. 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[单轮分析] 数据分割 | 训练集: {X_train.shape} | 测试集: {X_test.shape}")

    # 3. 模型训练与预测
    if model_name == "LGBM":
        model = get_lgbm_model(params=LGBM_PARAMS)
    elif model_name == "XGBoost":
        model = get_xgboost_model(params=XGBOOST_PARAMS)
    else:
        raise ValueError(f"不支持的模型: {model_name}，仅支持 'LGBM'/'XGBoost'")
    
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print(f"[单轮分析] {model_name}预测完成 | 训练集: {y_pred_train.shape} | 测试集: {y_pred_test.shape}")

    # 4. 构建结果表
    original_features = ["City Name", "Package", "Variety", "Origin", "Item Size", "Date"]

    # 训练集结果
    train_df = X_train.copy()
    train_df["真实价格"] = y_train.values
    train_df["预测价格"] = y_pred_train
    train_df["误差(预测-真实)"] = y_pred_train - y_train.values

    train_original = data.loc[X_train.index, original_features].copy()
    train_result = pd.concat([train_original, train_df], axis=1)

    # 测试集结果
    test_df = X_test.copy()
    test_df["真实价格"] = y_test.values
    test_df["预测价格"] = y_pred_test
    test_df["误差(预测-真实)"] = y_pred_test - y_test.values

    test_original = data.loc[X_test.index, original_features].copy()
    test_result = pd.concat([test_original, test_df], axis=1)

    # 5. 保存结果
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    train_save_path = os.path.join(output_dir, f"single_run_train_{model_name}.csv")
    test_save_path = os.path.join(output_dir, f"single_run_test_{model_name}.csv")
    
    train_result.to_csv(train_save_path, index=False)
    test_result.to_csv(test_save_path, index=False)
    print(f"[单轮分析] 训练集保存至: {train_save_path}")
    print(f"[单轮分析] 测试集保存至: {test_save_path}")

    # 6. 打印示例（与图片格式完全对齐）
    print("\n[单轮分析] 训练集部分样本（格式对齐）:")
    print(train_result[original_features + ["真实价格", "预测价格", "误差(预测-真实)"]].head(8))

    print("\n[单轮分析] 测试集部分样本（格式对齐）:")
    print(test_result[original_features + ["真实价格", "预测价格", "误差(预测-真实)"]].head(8))


if __name__ == "__main__":
    from data_analysis import load_data
    from configuration import DATA_PATH, FEATURE_COLUMNS, OUTPUT_CSV_PATH
    
    data = load_data()
    # 测试LGBM（单独运行脚本时默认）
    run_single_analysis(
        data=data,
        feature_columns=FEATURE_COLUMNS,
        model_name="LGBM",
        output_path=OUTPUT_CSV_PATH
    )
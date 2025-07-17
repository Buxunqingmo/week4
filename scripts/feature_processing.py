import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from configuration import TARGET_COLUMN

def process_features(df, feature_columns):
    """处理特征：日期解析、分类编码、缺失值填充"""
    # 复制数据避免修改原始数据
    df_processed = df[feature_columns + [TARGET_COLUMN]].copy()
    
    # 1. 处理日期特征（提取年份和月份）
    df_processed["Date"] = pd.to_datetime(df_processed["Date"])
    df_processed["year"] = df_processed["Date"].dt.year
    df_processed["month"] = df_processed["Date"].dt.month
    df_processed.drop("Date", axis=1, inplace=True)  # 丢弃原始日期列
    
    # 2. 填充缺失值（分类特征用众数，数值特征用均值）
    for col in df_processed.columns:
        if df_processed[col].dtype == "object":
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        else:
            df_processed[col].fillna(df_processed[col].mean(), inplace=True)
    
    # 3. 编码分类特征（ordinal encoding适合树模型）
    cat_cols = df_processed.select_dtypes(include="object").columns.tolist()
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_processed[cat_cols] = encoder.fit_transform(df_processed[cat_cols])
    
    # 分离特征和目标
    X = df_processed.drop(TARGET_COLUMN, axis=1)
    y = df_processed[TARGET_COLUMN]
    
    return X, y, encoder

if __name__ == "__main__":
    # 单元测试：特征处理
    from data_analysis import load_data
    data = load_data()
    X, y, encoder = process_features(data, FEATURE_COLUMNS)
    print("处理后特征形状:", X.shape)
    print("目标变量形状:", y.shape)
import pandas as pd
from configuration import DATA_PATH, TARGET_COLUMN
from configuration import FEATURE_COLUMNS, TARGET_COLUMN
def load_data():
    """加载数据并计算目标变量mean_price"""
    df = pd.read_csv(DATA_PATH)
    
    # 计算平均价格（目标变量）
    df[TARGET_COLUMN] = (df["Low Price"] + df["High Price"]) / 2
    
    # 查看基本信息
    print("数据形状:", df.shape)
    print("缺失值统计:\n", df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum())
    print("目标变量统计:\n", df[TARGET_COLUMN].describe())
    
    return df

if __name__ == "__main__":
    # 单元测试：加载数据
    data = load_data()
    print(data.head())
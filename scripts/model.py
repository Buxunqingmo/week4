import lightgbm as lgb
import xgboost as xgb
from configuration import LGBM_PARAMS, XGBOOST_PARAMS

def get_lgbm_model(params=LGBM_PARAMS):
    """初始化LGBM回归模型"""
    return lgb.LGBMRegressor(**params)

def get_xgboost_model(params=XGBOOST_PARAMS):
    """初始化XGBoost回归模型"""
    return xgb.XGBRegressor(** params)

def train_model(model, X_train, y_train):
    """训练模型"""
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # 单元测试：模型初始化
    lgbm = get_lgbm_model()
    xgb_model = get_xgboost_model()
    print("LGBM模型:", lgbm)
    print("XGBoost模型:", xgb_model)
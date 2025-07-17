import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from configuration import CV_FOLDS

def evaluate_performance(y_true, y_pred):
    """计算评估指标"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": round(rmse, 2), "mae": round(mae, 2), "r2": round(r2, 2)}

def cross_validate(model, X, y):
    """交叉验证并返回结果"""
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_results = {
        "fold_train": [],
        "fold_test": [],
        "average_train": None,
        "average_test": None
    }
    
    train_metrics = []
    test_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 训练与预测
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # 评估
        train_perf = evaluate_performance(y_train, y_train_pred)
        test_perf = evaluate_performance(y_test, y_test_pred)
        
        cv_results["fold_train"].append({
            "data_size": len(X_train),
            "performance": train_perf
        })
        cv_results["fold_test"].append({
            "data_size": len(X_test),
            "performance": test_perf
        })
        
        train_metrics.append(train_perf)
        test_metrics.append(test_perf)
    
    # 计算平均性能
    cv_results["average_train"] = {
        "rmse": round(np.mean([m["rmse"] for m in train_metrics]), 2),
        "mae": round(np.mean([m["mae"] for m in train_metrics]), 2),
        "r2": round(np.mean([m["r2"] for m in train_metrics]), 2)
    }
    cv_results["average_test"] = {
        "rmse": round(np.mean([m["rmse"] for m in test_metrics]), 2),
        "mae": round(np.mean([m["mae"] for m in test_metrics]), 2),
        "r2": round(np.mean([m["r2"] for m in test_metrics]), 2)
    }
    
    return cv_results

if __name__ == "__main__":
    # 单元测试：评估函数
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 33])
    print("评估结果:", evaluate_performance(y_true, y_pred))
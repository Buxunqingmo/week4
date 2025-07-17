import json
import os
from configuration import OUTPUT_JSON_PATH

def save_experiment_results(model_name, model_params, fea_encoding, cv_results):
    """保存实验结果到JSON文件"""
    result = {
        "model_name": model_name,
        "model_params": model_params,
        "fea_encoding": fea_encoding,
        "cv_results": cv_results
    }
    
    # 读取已有结果（如果存在）
    if os.path.exists(OUTPUT_JSON_PATH):
        with open(OUTPUT_JSON_PATH, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    all_results.append(result)
    
    # 保存更新后的结果
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"结果已保存至 {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    # 单元测试：保存结果
    test_result = {
        "model_name": "TestModel",
        "model_params": {"max_depth": 3},
        "fea_encoding": "ordinal",
        "cv_results": {"average_test": {"rmse": 10.0}}
    }
    save_experiment_results(**test_result)
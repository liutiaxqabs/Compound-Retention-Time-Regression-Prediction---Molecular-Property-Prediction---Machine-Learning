import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
import math
import pickle
import os

# 函数：读取CSV文件并返回其内容
def read_csv(filename):
    return pd.read_csv(filename)

# 函数：处理化合物的SMILES表示并计算描述符
def process_descriptors(data, include_rt=True):
    calc = Calculator(descriptors, ignore_3D=True)
    descriptors_list = []
    for smiles in data['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        res = calc(mol)
        descriptors_list.append([float(e) if not math.isnan(e) else 0 for e in res.values()])
    
    descriptor_names = [str(d) for d in calc.descriptors]
    descriptor_df = pd.DataFrame(descriptors_list, columns=descriptor_names)
    if include_rt and 'rt' in data.columns:
        descriptor_df['rt'] = data['rt']
    return descriptor_df

# 函数：训练模型并计算平均 MAE，保存模型和参数，time_int和cv_int代表实验次数和交叉验证倍数
def train_model(X, y, model, param_grid, model_name, folder_path, time_int=2, cv_int=10, random_state=42):
    mae_sum = 0
    best_mae = float('inf')
    best_model = None
    best_params = None

    for _ in range(time_int):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        grid_search = GridSearchCV(model, param_grid, cv=cv_int, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        y_hat = grid_search.predict(X_test)
        current_mae = mean_absolute_error(y_test, y_hat)
        mae_sum += current_mae

        if current_mae < best_mae:
            best_mae = current_mae
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

    avg_mae = mae_sum / time_int
    model_path = os.path.join(folder_path, f"{model_name}_best_model.pickle")
    params_path = os.path.join(folder_path, "model_params.txt")
    
    if best_model:
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        with open(params_path, 'a') as f:
            f.write(f"{model_name} Best Params: {best_params}\n")

    return avg_mae, model_path

# 函数：使用模型进行预测并保存预测结果
def predict_model(X, model_path, predictions_file):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    predictions = model.predict(X)
    np.savetxt(predictions_file, predictions, fmt='%f')
    return predictions

# 示例用法
def run_example():
    folder_path = "C:/Users/14980/Desktop/rt"  # 模型和参数的保存路径

    # 读取和处理训练数据
    path_csv = "C:/Users/14980/Desktop/rt/测试样例.csv"
    data = read_csv(path_csv)
    df_descriptors = process_descriptors(data, include_rt=True)
    X_train = df_descriptors.drop('rt', axis=1)
    y_train = df_descriptors['rt']

    # 定义模型和参数，参数来源于19年的”7个算法36个数据集“一篇综述，具体记不清楚了
    model_params = {
        "RandomForest": (
        RandomForestRegressor(), {"n_estimators": [100, 150, 200], "max_features": ['auto'], "max_depth": [50]}),
        "GradientBoosting": (GradientBoostingRegressor(),
                             {'n_estimators': [200, 300, 400, 500], 'max_depth': [2, 4],
                              'learning_rate': [0.01, 0.05, 0.1]}),
        "SVR": (SVR(), {'C': [50, 100, 150, 200, 250], 'gamma': [0.01, 1e-3, 1e-4], 'kernel': ['rbf']}),
        "Lasso": (Lasso(), {'alpha': [0.1, 0.5, 1, 5, 10]}),
        "MLPRegressor": (MLPRegressor(), {'alpha': [0.01], 'solver': ['adam'], 'max_iter': [2000],
                                          'hidden_layer_sizes': [(800, 400), (800, 400, 100)]})
    }

# 训练模型并保存结果
    results = {}
    for model_name, (model, params) in model_params.items():
        avg_mae, model_path = train_model(X_train, y_train, model, params, model_name, folder_path)
        results[model_name] = {"Average MAE": avg_mae, "Model Path": model_path}
    print(results)
    score_file = os.path.join(folder_path, "MAE_score.txt")
    # 提取模型名称和MAE指数，创建一个新的字典  
    model_mae_dict = {model: data['Average MAE'] for model, data in results.items()}  
    # 将字典写入文本文件  
    with open(score_file, 'w') as file:  
        for model, mae in model_mae_dict.items():  
            file.write(f'{model}: {mae}\n')

    # 预测示例：使用梯度提升树模型进行预测
    path_predict_csv = "C:/Users/14980/Desktop/rt/测试样例.csv"
    predict_data = read_csv(path_predict_csv)
    X_predict = process_descriptors(predict_data, include_rt=False)
    
    predictions_file = os.path.join(folder_path, "GradientBoosting_predictions.txt")
    rf_predictions = predict_model(X_predict, results["GradientBoosting"]["Model Path"], predictions_file)
    print("GradientBoosting Predictions saved to:", predictions_file)

# 主函数
if __name__ == "__main__":
    run_example()

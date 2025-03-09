import time
import numpy as np
import pandas as pd

# Progress bar
from tqdm import tqdm

# Models & Utilities
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# Preprocessing & Feature Selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# --- 1. Load Data (Assume 'data' is a pandas DataFrame with 'y' as the target) ---
print('pd.read_csv("data/foo.csv")')
data = pd.read_csv("data/foo.csv")
# For the example, we'll assume 'data' already exists in your environment.

# --- 2. Split into Features (X) and Target (y) ---
X = data.drop(columns=['y'])
y = data['y'].values

# --- 3. Define Feature-Selection Approaches ---
# We'll pick the top 100 features using two different strategies.

N_FEATURES = 100

# 3A. Filter-based selection: SelectKBest with f_regression
filter_selector = SelectKBest(score_func=f_regression, k=min(N_FEATURES, X.shape[1]))
print('X_kbest = filter_selector.fit_transform(X, y)')
X_kbest = filter_selector.fit_transform(X, y)

# 3B. Tree-based selection: RandomForest to get feature importances
rf_for_importances = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
print('rf_for_importances.fit(X, y)')
rf_for_importances.fit(X, y)

# Sort features by importance (descending order) and pick top N
importances = rf_for_importances.feature_importances_
feat_indices_sorted = np.argsort(importances)[::-1]  # descending
top_indices = feat_indices_sorted[:min(N_FEATURES, X.shape[1])]
X_treeSelected = X.iloc[:, top_indices]

# Make a dictionary of different feature sets
feature_sets = {
    "KBest_f_regression": X_kbest,
    "RF_Importances": X_treeSelected
}

# --- 4. Define Models ---
models = {
    "LinearRegression": LinearRegression(),
    "SVR_RBF": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    "MLPRegressor": MLPRegressor(hidden_layer_sizes=(2048,), max_iter=1000, early_stopping=True, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=50, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "XGBoost": XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
}

# --- 5. Define Multiple Metrics ---
# Note that cross_val_score only accepts one scoring metric at a time.
# We'll manually compute multiple metrics for each model via a custom function.

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_model(model, X_data, y_data, cv_splits=5):
    """
    Performs K-fold cross-validation for multiple metrics and returns the mean scores.
    """
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    
    for train_index, test_index in kf.split(X_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2_scores.append(r2_score(y_test, y_pred))
        #rmse_scores.append(mean_squared_error(y_test, y_pred, squared=False))
        mse = mean_squared_error(y_test, y_pred)
        rmse_scores.append(np.sqrt(mse))  # Manually take the square root
        mae_scores.append(mean_absolute_error(y_test, y_pred))
    
    return {
        "R2": np.mean(r2_scores),
        "RMSE": np.mean(rmse_scores),
        "MAE": np.mean(mae_scores)
    }

# --- 6. Main Loop Over Feature Sets and Models ---
# We'll keep track of results in a list of dicts, then convert to DataFrame.

all_results = []
start_time = time.time()

for fs_name, X_fs in feature_sets.items():
    print(f"\n=== Feature Selection: {fs_name} ===")
    
    # Convert to numpy array if it's a DataFrame
    if not isinstance(X_fs, np.ndarray):
        X_fs = X_fs.values
    
    # Scale the selected feature set
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fs)
    
    # Loop over models with a progress bar
    for model_name, model in tqdm(models.items(), desc=f"{fs_name} Models", ncols=80):
        scores = evaluate_model(model, X_scaled, y, cv_splits=5)
        result_dict = {
            "FeatureSet": fs_name,
            "Model": model_name,
            "R2": scores["R2"],
            "RMSE": scores["RMSE"],
            "MAE": scores["MAE"]
        }
        all_results.append(result_dict)

end_time = time.time()
elapsed = end_time - start_time

# --- 7. Convert Results to DataFrame & Display ---
results_df = pd.DataFrame(all_results)
# Sort by R2 descending
results_df = results_df.sort_values(by="R2", ascending=False).reset_index(drop=True)

print("\n*** MODEL RESULTS ***")
print(results_df)

print(f"\nTotal runtime: {elapsed:.2f} seconds (~{elapsed/60:.2f} minutes)")


#             FeatureSet             Model        R2      RMSE       MAE
# 0       RF_Importances  GradientBoosting  0.229424  4.998699  3.961131
# 1       RF_Importances      RandomForest  0.217356  5.038114  4.000649
# 2   KBest_f_regression  GradientBoosting  0.183821  5.146341  4.066487
# 3   KBest_f_regression      RandomForest  0.166679  5.198773  4.115009
# 4       RF_Importances           SVR_RBF  0.160846  5.217942  4.091167
# 5       RF_Importances           XGBoost  0.156713  5.228334  4.135610
# 6   KBest_f_regression           SVR_RBF  0.132727  5.304975  4.172001
# 7   KBest_f_regression  LinearRegression  0.119903  5.346688  4.215753
# 8       RF_Importances               KNN  0.101321  5.395694  4.279819
# 9   KBest_f_regression           XGBoost  0.081626  5.455615  4.301017
# 10  KBest_f_regression               KNN  0.070692  5.490196  4.412893
# 11      RF_Importances  LinearRegression  0.063038  5.514078  4.299393
# 12  KBest_f_regression      MLPRegressor -0.453318  6.841595  5.083355
# 13      RF_Importances      MLPRegressor -0.891840  7.809812  5.600293

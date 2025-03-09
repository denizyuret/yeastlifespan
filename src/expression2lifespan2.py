import time
import numpy as np
import pandas as pd

# Progress bar
from tqdm import tqdm

# Models & Utilities
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

# Preprocessing & Feature Selection
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import mutual_info_regression

# Evaluation Metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Suppress warnings for cleaner output
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- 1. Load Data ---
# Replace 'your_data.csv' with the path to your CSV file
print('pd.read_csv("data/foo.csv")')
data = pd.read_csv("data/foo.csv")

# --- 2. Split into Features (X) and Target (y) ---
X = data.drop(columns=['y'])
y = data['y'].values

# --- 3. Define Feature-Selection Approaches ---
# We will use multiple feature selection strategies

N_FEATURES = 100  # Number of top features to select

feature_sets = {}

# 3A. Filter-based selection: SelectKBest with f_regression
selector_kbest = SelectKBest(score_func=f_regression, k=min(N_FEATURES, X.shape[1]))
print('X_kbest = selector_kbest.fit_transform(X, y)')
X_kbest = selector_kbest.fit_transform(X, y)
feature_sets["SelectKBest_f_regression"] = X_kbest

# 3B. Filter-based selection: SelectKBest with mutual_info_regression
selector_mi = SelectKBest(score_func=mutual_info_regression, k=min(N_FEATURES, X.shape[1]))
print('X_mutual_info = selector_mi.fit_transform(X, y)')
X_mutual_info = selector_mi.fit_transform(X, y)
feature_sets["SelectKBest_mutual_info"] = X_mutual_info

# 3C. Tree-based selection: RandomForest feature importances
rf_importance = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
print('rf_importance.fit(X, y)')
rf_importance.fit(X, y)
importances_rf = rf_importance.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1][:N_FEATURES]
X_rf_selected = X.iloc[:, indices_rf].values
feature_sets["RandomForest_Importances"] = X_rf_selected

# 3D. Regularized selection: Lasso feature selection
lasso_selector = Lasso(alpha=0.001, max_iter=10000, random_state=42)
print('lasso_selector.fit(X, y)')
lasso_selector.fit(X, y)
# Select features where coefficients are non-zero
mask = lasso_selector.coef_ != 0
selected_features_lasso = X.columns[mask][:N_FEATURES]
X_lasso_selected = X[selected_features_lasso].values
feature_sets["Lasso_Selection"] = X_lasso_selected

# --- 4. Define Models ---
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001, max_iter=10000),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
#   "BayesianRidge": BayesianRidge(), # causes segfault
    "SVR_RBF": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    "SVR_Linear": SVR(kernel="linear", C=1.0, epsilon=0.1),
    "MLPRegressor": MLPRegressor(hidden_layer_sizes=(512,512,), max_iter=1000, early_stopping=True, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=50, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "XGBoost": XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1, verbosity=0),
    "LightGBM": lgb.LGBMRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    "CatBoost": CatBoostRegressor(iterations=50, random_state=42, verbose=0)
}

# --- 5. Define Evaluation Function ---
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
        
        # Compute metrics
        r2_scores.append(r2_score(y_test, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        rmse_scores.append(np.sqrt(mse))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
    
    return {
        "R2": np.mean(r2_scores),
        "RMSE": np.mean(rmse_scores),
        "MAE": np.mean(mae_scores)
    }

# --- 6. Main Loop Over Feature Sets and Models ---
all_results = []
start_time = time.time()

# Iterate over each feature selection method
for fs_name, X_fs in feature_sets.items():
    print(f"\n=== Feature Selection: {fs_name} ===")
    
    # Scale the selected feature set
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fs)
    
    # Iterate over each model with a progress bar
    for model_name, model in tqdm(models.items(), desc=f"{fs_name} Models", ncols=100):
        scores = evaluate_model(model, X_scaled, y, cv_splits=5)
        result_dict = {
            "FeatureSet": fs_name,
            "Model": model_name,
            "R2": scores["R2"],
            "RMSE": scores["RMSE"],
            "MAE": scores["MAE"]
        }
        print(result_dict)
        all_results.append(result_dict)

end_time = time.time()
elapsed = end_time - start_time

# --- 7. Convert Results to DataFrame & Display ---
results_df = pd.DataFrame(all_results)
# Sort by R2 descending
results_df = results_df.sort_values(by="R2", ascending=False).reset_index(drop=True)

# Display the results
print("\n*** MODEL RESULTS ***")
print(results_df)

print(f"\nTotal runtime: {elapsed:.2f} seconds (~{elapsed/60:.2f} minutes)")


#                   FeatureSet             Model        R2      RMSE       MAE
# 0   RandomForest_Importances  GradientBoosting  0.229424  4.998699  3.961131
# 1   RandomForest_Importances      RandomForest  0.217356  5.038114  4.000649
# 2   RandomForest_Importances          LightGBM  0.202584  5.086460  4.049864
# 3   SelectKBest_f_regression  GradientBoosting  0.183821  5.146341  4.066487
# 4   SelectKBest_f_regression      RandomForest  0.166679  5.198773  4.115009
# 5   RandomForest_Importances           SVR_RBF  0.160846  5.217942  4.091167
# 6   RandomForest_Importances           XGBoost  0.156713  5.228334  4.135610
# 7   SelectKBest_f_regression        ElasticNet  0.138449  5.289959  4.186351
# 8   SelectKBest_f_regression           SVR_RBF  0.132727  5.304975  4.172001
# 9    SelectKBest_mutual_info      RandomForest  0.129373  5.317759  4.246467
# 10   SelectKBest_mutual_info  GradientBoosting  0.128465  5.317391  4.219763
# 11  SelectKBest_f_regression             Lasso  0.122386  5.339133  4.211854
# 12  SelectKBest_f_regression             Ridge  0.121772  5.341002  4.212893
# 13  SelectKBest_f_regression  LinearRegression  0.119903  5.346688  4.215753
# 14  SelectKBest_f_regression          LightGBM  0.118273  5.349008  4.212980
# 15   SelectKBest_mutual_info           SVR_RBF  0.117898  5.350566  4.211128
# 16           Lasso_Selection      RandomForest  0.114267  5.362041  4.254798
# 17           Lasso_Selection  GradientBoosting  0.112438  5.367061  4.259160
# 18  RandomForest_Importances               KNN  0.101321  5.395694  4.279819
# 19  RandomForest_Importances          CatBoost  0.101264  5.395450  4.260493
# 20   SelectKBest_mutual_info               KNN  0.095037  5.413534  4.286193
# 21   SelectKBest_mutual_info          LightGBM  0.090886  5.431109  4.324394
# 22  RandomForest_Importances        ElasticNet  0.088637  5.438359  4.261644
# 23           Lasso_Selection           SVR_RBF  0.088067  5.439587  4.270223
# 24           Lasso_Selection          LightGBM  0.084382  5.451791  4.335654
# 25  SelectKBest_f_regression           XGBoost  0.081626  5.455615  4.301017
# 26  SelectKBest_f_regression        SVR_Linear  0.080091  5.464984  4.298042
# 27  SelectKBest_f_regression               KNN  0.070692  5.490196  4.412893
# 28  RandomForest_Importances             Lasso  0.066875  5.502824  4.293649
# 29  RandomForest_Importances             Ridge  0.064964  5.508393  4.296833
# 30  RandomForest_Importances  LinearRegression  0.063038  5.514078  4.299393
# 31  SelectKBest_f_regression          CatBoost  0.052961  5.533034  4.370987
# 32   SelectKBest_mutual_info        ElasticNet  0.039506  5.583466  4.349184
# 33   SelectKBest_mutual_info        SVR_Linear  0.036511  5.595439  4.337690
# 34           Lasso_Selection               KNN  0.036503  5.584261  4.412217
# 35           Lasso_Selection           XGBoost  0.025694  5.619224  4.451017
# 36   SelectKBest_mutual_info             Lasso  0.019070  5.642156  4.386258
# 37   SelectKBest_mutual_info             Ridge  0.017661  5.646215  4.388758
# 38   SelectKBest_mutual_info  LinearRegression  0.015724  5.651720  4.392067
# 39  RandomForest_Importances        SVR_Linear  0.008619  5.666491  4.389504
# 40   SelectKBest_mutual_info           XGBoost -0.009256  5.722491  4.512829
# 41   SelectKBest_mutual_info          CatBoost -0.020839  5.752239  4.506470
# 42           Lasso_Selection          CatBoost -0.024875  5.767782  4.596251
# 43           Lasso_Selection        ElasticNet -0.040672  5.806355  4.437879
# 44           Lasso_Selection             Lasso -0.076718  5.903823  4.474887
# 45           Lasso_Selection             Ridge -0.078793  5.909432  4.477769
# 46           Lasso_Selection  LinearRegression -0.082143  5.918356  4.480985
# 47           Lasso_Selection        SVR_Linear -0.183198  6.187044  4.588824
# 48  SelectKBest_f_regression      MLPRegressor -0.388267  6.680344  5.040116
# 49   SelectKBest_mutual_info      MLPRegressor -0.938700  7.894375  5.751250
# 50  RandomForest_Importances      MLPRegressor -1.045688  8.106889  5.560205
# 51           Lasso_Selection      MLPRegressor -1.824152  9.542614  6.590434

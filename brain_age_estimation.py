import pandas as pd
import numpy as np
import time
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR 
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

class RegressionModelTrainer:
    def __init__(self, X, y, results_dir, harmonization_strategy, dataset, stratify_cols=None):
        self.X = X
        self.y = y
        self.results_dir = results_dir
        self.harmonization_strategy = harmonization_strategy
        self.dataset = dataset
        self.stratify_cols = stratify_cols
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Support Vector Regression': SVR(kernel='rbf'),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
            # 'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            # 'Hist Gradient Boosting': HistGradientBoostingRegressor(max_iter=100, random_state=42),
            # 'XGBoost': xgb.XGBRegressor(n_estimators=100, n_jobs=-1, random_state=42, verbosity=0)
        }
        self.metrics = ['MAE', 'RMSE', 'R2 Score', 'Runtime']

    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        start = time.time()
        model.fit(X_train, y_train)
        stop = time.time()
        time_new = stop - start

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_metrics = self.calculate_metrics(y_train, y_pred_train, time_new)
        test_metrics = self.calculate_metrics(y_test, y_pred_test, time_new)

        return train_metrics + test_metrics

    def calculate_metrics(self, y_true, y_pred, runtime):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_squared_error(y_true, y_pred)
        return [mae, rmse, r2, runtime]

    def train_and_evaluate(self, n_splits=5, stratify=False):
        if stratify:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            stratify_labels = self.stratify_cols
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            stratify_labels = None

        results = {model_name: [] for model_name in self.models}

        for fold, (train_index, test_index) in enumerate(kf.split(self.X, stratify_labels)):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            for model_name, model in self.models.items():
                metrics = self.evaluate_model(model, X_train, y_train, X_test, y_test)
                results[model_name].append([fold] + metrics)

        return results

    def save_results(self, results, mean_df, full_df, mean_std_df, stratify_strategy):
        mean_results = {model_name: np.mean(metrics, axis=0) for model_name, metrics in results.items()}
        std_results = {model_name: np.std(metrics, axis=0) for model_name, metrics in results.items()}
        full_results = {model_name: metrics for model_name, metrics in results.items()}

        mean_std_results = {model_name: [f"{mean:.3f} ± {std:.3f}" for mean, std in zip(mean_results[model_name], std_results[model_name])] for model_name in mean_results}

        mean_df_temp = pd.DataFrame(mean_results, index=['Fold'] + self.metrics * 2).T
        full_df_temp = pd.DataFrame([item for sublist in full_results.values() for item in sublist],
                                    columns=['Fold'] + self.metrics * 2)
        mean_std_df_temp = pd.DataFrame(mean_std_results, index=['Fold'] + self.metrics * 2).T

        # Set the index of full_df_temp to the model names
        full_df_temp.index = [model_name for model_name, metrics in full_results.items() for _ in metrics]

        mean_df_temp['Harmonization Strategy'] = self.harmonization_strategy
        mean_df_temp['Stratification Strategy'] = stratify_strategy
        full_df_temp['Harmonization Strategy'] = self.harmonization_strategy
        full_df_temp['Stratification Strategy'] = stratify_strategy
        mean_std_df_temp['Harmonization Strategy'] = self.harmonization_strategy
        mean_std_df_temp['Stratification Strategy'] = stratify_strategy

        mean_df = pd.concat([mean_df, mean_df_temp])
        full_df = pd.concat([full_df, full_df_temp])
        mean_std_df = pd.concat([mean_std_df, mean_std_df_temp])

        return mean_df, full_df, mean_std_df

    def run(self, mean_df, full_df, mean_std_df, stratify=False):
        stratify_strategy = 'Stratified' if stratify else 'Random'
        results = self.train_and_evaluate(stratify=stratify)
        mean_df, full_df, mean_std_df = self.save_results(results, mean_df, full_df, mean_std_df, stratify_strategy)
        return mean_df, full_df, mean_std_df

import os
import numpy as np
import pandas as pd
import timeit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, balanced_accuracy_score
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

class ClassificationModelTrainer:
    def __init__(self, X, y, results_dir, harmonization_strategy, dataset, stratify_cols=None):
        self.X = X
        self.y = y
        self.results_dir = results_dir
        self.harmonization_strategy = harmonization_strategy
        self.dataset = dataset
        self.stratify_cols = stratify_cols
        self.models = {
            'SVM': self.svm_fun,
            'NB': self.gaus_nb_fun
            # 'MLP': self.mlp_fun,
            # 'KNN': self.knn_fun,
            # 'RF': self.rf_fun,
            # 'LR': self.lr_fun,
            # 'DT': self.fun_decision_tree
        }
        self.metrics = ['Accuracy', 'Precision', 'Recall', 'F1 (weighted)', 'F1 (Macro)', 'ROC AUC', 'Balanced Accuracy', 'Runtime']

    def roc_auc_score_multiclass(self, actual_class, pred_class, average="macro"):
        unique_class = set(actual_class)
        roc_auc_dict = {}
        for per_class in unique_class:
            other_class = [x for x in unique_class if x != per_class]
            new_actual_class = [0 if x in other_class else 1 for x in actual_class]
            new_pred_class = [0 if x in other_class else 1 for x in pred_class]
            roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
            roc_auc_dict[per_class] = roc_auc
        return np.mean(list(roc_auc_dict.values()))

    def evaluate_model(self, model_func, X_train, y_train, X_test, y_test):
        start = timeit.default_timer()
        model_func(X_train, y_train, X_test, y_test)
        stop = timeit.default_timer()
        time_new = stop - start

        y_pred_train = model_func(X_train, y_train, X_train, y_train)
        y_pred_test = model_func(X_train, y_train, X_test, y_test)

        train_metrics = self.calculate_metrics(y_train, y_pred_train, time_new)
        test_metrics = self.calculate_metrics(y_test, y_pred_test, time_new)

        return train_metrics + test_metrics

    def calculate_metrics(self, y_true, y_pred, runtime):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        roc_auc = self.roc_auc_score_multiclass(y_true, y_pred, average='macro')
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        return [accuracy, precision, recall, f1_weighted, f1_macro, roc_auc, balanced_acc, runtime]

    def svm_fun(self, X_train, y_train, X_test, y_test):
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)
        return clf.predict(X_test)

    def gaus_nb_fun(self, X_train, y_train, X_test, y_test):
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        return gnb.predict(X_test)

    def mlp_fun(self, X_train, y_train, X_test, y_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
        mlp.fit(X_train, y_train)
        return mlp.predict(X_test)

    def knn_fun(self, X_train, y_train, X_test, y_test):
        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(X_train, y_train)
        return classifier.predict(X_test)

    def rf_fun(self, X_train, y_train, X_test, y_test):
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train, y_train)
        return rf.predict(X_test)

    def lr_fun(self, X_train, y_train, X_test, y_test):
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)
        model = LogisticRegression(solver='liblinear', random_state=0)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    def fun_decision_tree(self, X_train, y_train, X_test, y_test):
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        return clf.predict(X_test)

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

            for model_name, model_func in self.models.items():
                metrics = self.evaluate_model(model_func, X_train, y_train, X_test, y_test)
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
        mean_std_df_temp = pd.DataFrame(mean_std_results, index=['Fold'] +self.metrics * 2).T  # .reset_index(drop=True)

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Classification Task')
    parser.add_argument('--X', type=str, required=True, help='Path to the features file')
    parser.add_argument('--y', type=str, required=True, help='Path to the labels file')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--harmonization_strategy', type=str, required=True, choices=['unharmonized', 'combat', 'covbat'], help='Harmonization strategy')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., desikan_roi, vbm_roi, destrieux_roi)')

    args = parser.parse_args()

    X = np.load(args.X)
    y = np.load(args.y)

    trainer = ClassificationModelTrainer(X, y, args.results_dir, args.harmonization_strategy, args.dataset)
    trainer.run()

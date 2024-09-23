import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('Data02_FI/data2.csv')
df_transposed = df.transpose()
df_transposed = df_transposed.drop(df_transposed.index[0])
last_column = df_transposed.iloc[:, -1]
encoder = LabelEncoder()
encoded_last_column = encoder.fit_transform(last_column)
df_transposed['Target'] = encoded_last_column
df_transposed = df_transposed.drop(df_transposed.columns[-2], axis=1)

X = df_transposed.drop(columns=["Target"])
y = df_transposed["Target"]

le = LabelEncoder()
for column in X.columns:
    if X[column].dtype == 'object':
        X[column] = le.fit_transform(X[column])

X = pd.get_dummies(X, columns=X.select_dtypes(include='category').columns)

def train_and_evaluate_xgboost(X, y, num_features):
    top_features = X.columns[:num_features]
    X_top_features = X[top_features]
    xgb_classifier = XGBClassifier()
    scores = cross_val_score(xgb_classifier, X_top_features, y, cv=5)
    return scores, np.mean(scores), np.std(scores)

def plot_xgboost_mean_accuracy_vs_features(X, y, num_features_range):
    mean_accuracies = []
    std_devs = []

    for num_features in num_features_range:
        fold_accuracies, mean_accuracy, std_dev = train_and_evaluate_xgboost(X, y, num_features)
        mean_accuracies.append(mean_accuracy)
        std_devs.append(std_dev)

        print(f"Number of Features: {num_features}")
        for i, acc in enumerate(fold_accuracies):
            print(f"Fold {i+1} - Accuracy: {acc:.2f}")
        print(f"Mean Accuracy: {mean_accuracy:.2f}")
        print(f"Standard Deviation: {std_dev:.2f}")
        print()

    results_df = pd.DataFrame({
        'Num Features': num_features_range,
        'Mean Accuracy': mean_accuracies,
        'Standard Deviation': std_devs
    })
    results_df.to_csv('xgboost_evaluation_results.csv', index=False)

    plt.errorbar(num_features_range, mean_accuracies, yerr=std_devs, fmt='o-', label='Mean Accuracy')
    plt.xlabel("Number of Features")
    plt.ylabel("Mean Accuracy")
    plt.title("XGBoost Mean Accuracy vs. Number of Features with Standard Deviation")
    plt.legend()
    plt.savefig('xgboost_mean_accuracy_vs_features.png')
    plt.show()

num_features_range = range(100, 170, 10)
plot_xgboost_mean_accuracy_vs_features(X, y, num_features_range)

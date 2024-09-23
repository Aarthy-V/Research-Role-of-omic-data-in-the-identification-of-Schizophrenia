import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score


df = pd.read_csv('Data01_FI/data1.csv')
df_transposed = df.transpose()
df_transposed = df_transposed.drop(df_transposed.index[0])
last_column = df_transposed.iloc[:, -1]
encoder = LabelEncoder()
encoded_last_column = encoder.fit_transform(last_column)
df_transposed['Target'] = encoded_last_column
df_transposed = df_transposed.drop(df_transposed.columns[60676], axis=1)

X = df_transposed.iloc[:, :-1] 
y = df_transposed['Target'] 
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X, y)
feature_importances = rf_classifier.feature_importances_
importance_series = pd.Series(feature_importances, index=X.columns, name="Feature Importances")
importance_series.to_csv('feature_importances.csv')

plt.figure(figsize=(20, 10))
importance_series.plot(kind='bar')
plt.title('Feature Importances')
plt.ylabel('Importance Score')
plt.xlabel('Features')
plt.xticks(rotation=90) 
plt.savefig('all_feature_importances_plot.png')
plt.show()

importance_threshold = 0.0
selected_features = importance_series[importance_series > importance_threshold].index.tolist()
X_selected = X[selected_features]


def train_and_evaluate_svm(X, y, num_features, n_folds):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=num_features)
    X_pca = pca.fit_transform(X_scaled)
    svm_classifier = SVC(kernel='linear', decision_function_shape='ovr')
    scores = cross_val_score(svm_classifier, X_pca, y, cv=n_folds)

    for i, acc in enumerate(scores):
        print(f"Fold {i + 1} - Accuracy: {acc:.2f}")

    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)
    print(f"Mean Accuracy ({num_features} features): {mean_accuracy:.2f}")
    print(f"Standard Deviation ({num_features} features): {std_accuracy:.2f}")
    print()

    return mean_accuracy, std_accuracy

def plot_svm_mean_accuracy_vs_features(X, y, num_features_range, n_folds):
    mean_accuracies = []
    std_accuracies = []

    for num_features in num_features_range:
        mean_accuracy, std_accuracy = train_and_evaluate_svm(X, y, num_features, n_folds)
        mean_accuracies.append(mean_accuracy)
        std_accuracies.append(std_accuracy)

    plt.figure(figsize=(10, 6))
    plt.errorbar(num_features_range, mean_accuracies, yerr=std_accuracies, fmt='o-', capsize=4)
    plt.xlabel("Number of Features")
    plt.ylabel("Mean Accuracy")
    plt.title("SVM Mean Accuracy (Linear Kernel) vs. Number of Features")
    plt.savefig('svm_mean_accuracy_vs_features.png')
    plt.show()

num_features_range = range(10, 200, 10) 
n_folds = 5 

plot_svm_mean_accuracy_vs_features(X_selected, y, num_features_range, n_folds)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score


df = pd.read_csv('Data02_FI/data2.csv')
df_transposed = df.transpose()
df_transposed = df_transposed.drop(df_transposed.index[0])
duplicates = df_transposed.duplicated()
print("Number of duplicate rows:", duplicates.sum())
missing_percentage = round((df_transposed.isnull().sum() / df_transposed.shape[0]) * 100, 2)
print("Percentage of missing values per column:\n", missing_percentage)
last_column = df_transposed.iloc[:, -1]
encoder = LabelEncoder()
encoded_last_column = encoder.fit_transform(last_column)
df_transposed['Target'] = encoded_last_column
df_transposed = df_transposed.drop(df_transposed.columns[-2], axis=1)

X = df_transposed.drop(columns=["Target"])
y = df_transposed["Target"]

target_sample_count = 80
class_counts = Counter(y)
desired_class_counts = {cls: target_sample_count for cls in class_counts.keys()}

ros = RandomOverSampler(sampling_strategy=desired_class_counts, random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
rus = RandomUnderSampler(sampling_strategy=desired_class_counts, random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_resampled, y_resampled)
feature_importances = rf_classifier.feature_importances_
importance_series = pd.Series(feature_importances, index=X_resampled.columns, name="Feature Importances")
importance_series = importance_series.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importance_series.plot(kind='bar')
plt.title("Feature Importances from RandomForest")
plt.ylabel("Importance")
plt.xlabel("Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

importance_threshold = 0.0 
selected_features = importance_series[importance_series > importance_threshold].index.tolist()
X_selected = X_resampled[selected_features]
print("Selected features:\n", X_selected.columns)

print("Final X_selected shape:", X_selected.shape)
print("Final y_resampled shape:", y_resampled.shape)


# Define a function to train and evaluate SVM with a linear kernel using 10-fold CV
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
    # Create a list to store mean accuracies and standard deviations for different numbers of features
    mean_accuracies = []
    std_accuracies = []

    for num_features in num_features_range:
        mean_accuracy, std_accuracy = train_and_evaluate_svm(X_selected, y, num_features, n_folds)
        mean_accuracies.append(mean_accuracy)
        std_accuracies.append(std_accuracy)

    results_df = pd.DataFrame({
        'Number of Features': num_features_range,
        'Mean Accuracy': mean_accuracies,
        'Standard Deviation': std_accuracies
    })
    results_df.to_csv('svm_accuracies.csv', index=False)

    plt.errorbar(num_features_range, mean_accuracies, yerr=std_accuracies, fmt='o-', capsize=4)
    plt.xlabel("Number of Features")
    plt.ylabel("Mean Accuracy")
    plt.title("SVM Mean Accuracy (Linear Kernel) vs. Number of Features")
    plt.savefig("svm_accuracy_plot.png")
    plt.show()
    
num_features_range = range(100, 150, 10)
n_folds = 5

plot_svm_mean_accuracy_vs_features(X_selected, y_resampled, num_features_range, n_folds)


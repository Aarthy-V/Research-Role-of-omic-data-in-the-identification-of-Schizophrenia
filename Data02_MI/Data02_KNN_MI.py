import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# Function to load and preprocess the dataset
def preprocess_data(file_path, target_sample_count=80, random_state=42):
    df = pd.read_csv(file_path)
    df_transposed = df.transpose()
    df_transposed = df_transposed.drop(df_transposed.index[0])
    last_column = df_transposed.iloc[:, -1]
    encoder = LabelEncoder()
    encoded_last_column = encoder.fit_transform(last_column)
    df_transposed['Target'] = encoded_last_column
    df_transposed = df_transposed.drop(df_transposed.columns[-2], axis=1)
    X = df_transposed.drop(columns=["Target"])
    y = df_transposed["Target"]

    class_counts = dict(y.value_counts())
    sampling_strategy = {cls: max(count, target_sample_count) for cls, count in class_counts.items()}
    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    rus = RandomUnderSampler(sampling_strategy={cls: target_sample_count for cls in class_counts.keys()}, random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)
    
    return X_resampled, y_resampled

# Function to calculate mutual information scores and perform feature selection
def feature_selection(X, y, mi_threshold=0.0):
    mi_scores = mutual_info_classif(X, y)
    mi_scores_series = pd.Series(mi_scores, index=X.columns, name="MI Scores")
    selected_features = mi_scores_series[mi_scores_series > mi_threshold].index.tolist()
    X_selected = X[selected_features]
    
    return X_selected, mi_scores_series

# Function to train and evaluate KNN with varied feature counts and k values
def train_and_evaluate_knn_with_varied_features(X, y, mi_scores_series, feature_counts, k_values):
    results = []

    for num_features in feature_counts:
        top_features = mi_scores_series.nlargest(num_features).index.tolist()
        X_top_features = X[top_features]

        for k in k_values:
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn_classifier, X_top_features, y, cv=5)
            mean_accuracy = scores.mean()
            results.append((num_features, k, mean_accuracy))

    return results

# Function to plot heatmap of KNN mean accuracy for different k values and number of features
def plot_knn_heatmap(results, filename="knn_heatmap.png"):
    result_df = pd.DataFrame(results, columns=["Num Features", "k Value", "Mean Accuracy"])
    heatmap_data = result_df.pivot(index="k Value", columns="Num Features", values="Mean Accuracy")

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("KNN Mean Accuracy Heatmap")
    plt.xlabel("Number of Features")
    plt.ylabel("k Value")
    plt.savefig(filename)

    # Find and print the best combination of features and k value
    best_result = result_df.loc[result_df["Mean Accuracy"].idxmax()]
    best_num_features, best_k, best_accuracy = best_result["Num Features"], best_result["k Value"], best_result["Mean Accuracy"]
    
    print(f"Best Number of Features: {best_num_features}")
    print(f"Best k Value: {best_k}")
    print(f"Best Accuracy: {best_accuracy:.2f}")

# Main function to run the KNN workflow
def main_knn(file_path):
    X_resampled, y_resampled = preprocess_data(file_path)
    X_selected, mi_scores_series = feature_selection(X_resampled, y_resampled, mi_threshold=0.0)
    feature_counts_to_test = list(range(50, 160, 10))
    k_values_to_test = list(range(3, 21))
    results = train_and_evaluate_knn_with_varied_features(X_selected, y_resampled, mi_scores_series, feature_counts_to_test, k_values_to_test)

    results_df = pd.DataFrame(results, columns=["Num Features", "k Value", "Mean Accuracy"])
    results_df.to_csv("knn_results.csv", index=False)

    plot_knn_heatmap(results, filename="knn_heatmap.png")

file_path = 'Data02/data2.csv' 
main_knn(file_path)

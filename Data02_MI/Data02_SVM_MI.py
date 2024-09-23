import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

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

# Function to train and evaluate SVM with cross-validation
def train_and_evaluate_svm(X, y, num_features, mi_scores_series):
    top_features = mi_scores_series.nlargest(num_features).index.tolist()
    X_top_features = X[top_features]
    svm_classifier = SVC(kernel='linear', decision_function_shape='ovr')
    scores = cross_val_score(svm_classifier, X_top_features, y, cv=5)

    mean_accuracy = scores.mean()
    std_accuracy = scores.std()
    return mean_accuracy, std_accuracy

# Function to plot mean accuracy vs number of features
def plot_svm_mean_accuracy_vs_features(X, y, mi_scores_series, num_features_range, filename="svm_mean_accuracy_vs_features.png"):
    mean_accuracies = []
    std_accuracies = []

    for num_features in num_features_range:
        mean_accuracy, std_accuracy = train_and_evaluate_svm(X, y, num_features, mi_scores_series)
        mean_accuracies.append(mean_accuracy)
        std_accuracies.append(std_accuracy)

    plt.errorbar(num_features_range, mean_accuracies, yerr=std_accuracies, fmt='o-', capsize=4)
    plt.xlabel("Number of Features")
    plt.ylabel("Mean Accuracy")
    plt.title("SVM Mean Accuracy (Linear Kernel) vs. Number of Features")
    plt.savefig(filename)
    plt.show()

# Main function to run the entire workflow
def main(file_path):
    X_resampled, y_resampled = preprocess_data(file_path)
    X_selected, mi_scores_series = feature_selection(X_resampled, y_resampled, mi_threshold=0.0)
    mi_scores_series.sort_values(ascending=False).to_csv("mi_scores.csv")

    print("Mutual Information Scores:")
    print(mi_scores_series.sort_values(ascending=False))
 
    num_features_range = range(50, 160, 10) 
    plot_svm_mean_accuracy_vs_features(X_selected, y_resampled, mi_scores_series, num_features_range, filename="svm_mean_accuracy_vs_features.png")


file_path = 'Data02/data2.csv'  
main(file_path)

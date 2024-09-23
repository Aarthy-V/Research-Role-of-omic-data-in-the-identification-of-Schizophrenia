import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
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
def feature_selection_rf(X, y, mi_threshold=0.0):
    mi_scores = mutual_info_classif(X, y)
    mi_scores_series = pd.Series(mi_scores, index=X.columns, name="MI Scores")
    selected_features = mi_scores_series[mi_scores_series > mi_threshold].index.tolist()
    X_selected = X[selected_features]
    
    return X_selected, mi_scores_series

# Function to train and evaluate Random Forest with cross-validation
def train_and_evaluate_rf(X, y, num_features, mi_scores_series, n_estimators=50):
    top_features = mi_scores_series.nlargest(num_features).index.tolist()
    X_top_features = X[top_features]
    rf_classifier = RandomForestClassifier(class_weight='balanced', n_estimators=n_estimators, random_state=42)
    scores = cross_val_score(rf_classifier, X_top_features, y, cv=5, scoring='accuracy')
    mean_accuracy = scores.mean()
    std_accuracy = scores.std()
    
    return mean_accuracy, std_accuracy, scores

# Function to plot mean accuracy vs number of features for Random Forest
def plot_rf_mean_accuracy_vs_features(X, y, mi_scores_series, num_features_range, n_estimators=50):
    mean_accuracies = []
    std_accuracies = []
    fold_accuracies = pd.DataFrame()

    for num_features in num_features_range:
        mean_accuracy, std_accuracy, scores = train_and_evaluate_rf(X, y, num_features, mi_scores_series, n_estimators)
        mean_accuracies.append(mean_accuracy)
        std_accuracies.append(std_accuracy)
        fold_accuracies[f'num_features_{num_features}'] = scores

    # Save fold accuracy results to CSV
    fold_accuracies.to_csv('fold_accuracies_rf.csv', index=False)
    
    # Plot mean accuracy vs number of features
    plt.errorbar(num_features_range, mean_accuracies, yerr=std_accuracies, fmt='o-', capsize=4)
    plt.xlabel("Number of Features")
    plt.ylabel("Mean Accuracy")
    plt.title(f"Random Forest Mean Accuracy vs. Number of Features (n_estimators={n_estimators})")
    plt.savefig('rf_accuracy_vs_features.png')
    plt.show()

def main_rf(file_path, n_estimators=50):
    X_resampled, y_resampled = preprocess_data(file_path)
    X_selected, mi_scores_series = feature_selection_rf(X_resampled, y_resampled, mi_threshold=0.0)
    mi_scores_series.sort_values(ascending=False).to_csv("mi_scores.csv")
    print("Mutual Information Scores (Random Forest):")
    print(mi_scores_series.sort_values(ascending=False))
    num_features_range = range(50, 160, 10)  # Adjust as needed

    plot_rf_mean_accuracy_vs_features(X_selected, y_resampled, mi_scores_series, num_features_range, n_estimators)

file_path = 'Data02/data2.csv' 
main_rf(file_path)

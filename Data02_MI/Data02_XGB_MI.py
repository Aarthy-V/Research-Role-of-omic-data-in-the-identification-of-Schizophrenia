import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
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
    
    # Convert all columns to numeric, forcing errors to NaN (e.g., strings will become NaN)
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Drop columns that have become all NaN after conversion
    X = X.dropna(axis=1, how='all')
    
    class_counts = dict(y.value_counts())
    sampling_strategy = {cls: max(count, target_sample_count) for cls, count in class_counts.items()}

    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    rus = RandomUnderSampler(sampling_strategy={cls: target_sample_count for cls in class_counts.keys()}, random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)
    
    return X_resampled, y_resampled

# Function to calculate mutual information scores and perform feature selection
def feature_selection_xgb(X, y, mi_threshold=0.0):
    mi_scores = mutual_info_classif(X, y)
    mi_scores_series = pd.Series(mi_scores, index=X.columns, name="MI Scores")
    selected_features = mi_scores_series[mi_scores_series > mi_threshold].index.tolist()
    X_selected = X[selected_features]
    
    return X_selected, mi_scores_series

# Function to train and evaluate XGBoost with cross-validation
def train_and_evaluate_xgb(X, y, num_features, mi_scores_series, n_estimators=50):
    top_features = mi_scores_series.nlargest(num_features).index.tolist()
    X_top_features = X[top_features]
    xgb_classifier = XGBClassifier(n_estimators=n_estimators, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    scores = cross_val_score(xgb_classifier, X_top_features, y, cv=5, scoring='accuracy')
    mean_accuracy = scores.mean()
    std_accuracy = scores.std()
    
    return mean_accuracy, std_accuracy

# Function to plot mean accuracy vs number of features for XGBoost and save accuracies
def plot_xgb_mean_accuracy_vs_features(X, y, mi_scores_series, num_features_range, n_estimators=50):
    mean_accuracies = []
    std_accuracies = []

    for num_features in num_features_range:
        mean_accuracy, std_accuracy = train_and_evaluate_xgb(X, y, num_features, mi_scores_series, n_estimators)
        mean_accuracies.append(mean_accuracy)
        std_accuracies.append(std_accuracy)

    plt.errorbar(num_features_range, mean_accuracies, yerr=std_accuracies, fmt='o-', capsize=4)
    plt.xlabel("Number of Features")
    plt.ylabel("Mean Accuracy")
    plt.title(f"XGBoost Mean Accuracy vs. Number of Features (n_estimators={n_estimators})")
    plt.savefig("XGBoost_accuracy_graph.png")
    plt.show()

    # Save the accuracies to a CSV file
    accuracy_df = pd.DataFrame({
        'Num_Features': num_features_range,
        'Mean_Accuracy': mean_accuracies,
        'Std_Accuracy': std_accuracies
    })
    accuracy_df.to_csv("xgboost_accuracies.csv", index=False)

# Main function to run the XGBoost workflow
def main_xgb(file_path, n_estimators=50):
    X_resampled, y_resampled = preprocess_data(file_path)
    X_selected, mi_scores_series = feature_selection_xgb(X_resampled, y_resampled, mi_threshold=0.0)
    mi_scores_series.sort_values(ascending=False).to_csv("mi_scores_xgb.csv")
    print("Mutual Information Scores (XGBoost):")
    print(mi_scores_series.sort_values(ascending=False))
    num_features_range = range(50, 160, 10) 

    plot_xgb_mean_accuracy_vs_features(X_selected, y_resampled, mi_scores_series, num_features_range, n_estimators)

file_path = 'Data02_MI/data2.csv' 
main_xgb(file_path)

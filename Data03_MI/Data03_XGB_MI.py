import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import csv

# Load and preprocess the dataset
df = pd.read_csv('Data03_MI/top_1200_featuresN.csv')
df.duplicated()
round((df.isnull().sum() / df.shape[0]) * 100, 2)
last_column = df.iloc[:, -1]
encoder = LabelEncoder()
encoded_last_column = encoder.fit_transform(last_column)
df['Target'] = encoded_last_column
df = df.drop(df.columns[1200], axis=1)

X = df.drop(columns=["Target"])
y = df["Target"]

# Calculate mutual information scores
mi_scores = mutual_info_classif(X, y)
mi_scores_series = pd.Series(mi_scores, index=X.columns, name="MI Scores")
mi_scores_series = mi_scores_series.sort_values(ascending=False)
mi_threshold = 0  
selected_features = mi_scores_series[mi_scores_series > mi_threshold].index.tolist()
X_selected = X[selected_features]

# Function to train and evaluate XGBoost with cross-validation
def train_and_evaluate_xgb(X, y, num_features, n_folds):
    # Select top features
    top_features = mi_scores_series.nlargest(num_features).index.tolist()
    X_top_features = X[top_features]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_top_features)
    
    # Initialize XGBoost classifier
    xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    
    # Cross-validation
    scores = cross_val_score(xgb_classifier, X_scaled, y, cv=n_folds, scoring='accuracy')
    fold_results = []
    for i, acc in enumerate(scores):
        fold_results.append((num_features, f"Fold {i + 1}", acc))

    return fold_results

# Function to plot mean accuracy vs number of features for XGBoost and save accuracies
def plot_xgb_mean_accuracy_vs_features(X, y, mi_scores_series, num_features_range, n_folds, output_csv):
    all_fold_results = []
    
    mean_accuracies = []
    std_accuracies = []

    for num_features in num_features_range:
        fold_results = train_and_evaluate_xgb(X, y, num_features, n_folds)
        all_fold_results.extend(fold_results)

        fold_accuracies = [acc for _, _, acc in fold_results]
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        mean_accuracies.append(mean_accuracy)
        std_accuracies.append(std_accuracy)

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Number of Features", "Fold", "Accuracy"])
        writer.writerows(all_fold_results)

    plt.errorbar(num_features_range, mean_accuracies, yerr=std_accuracies, fmt='o-', capsize=4)
    plt.xlabel("Number of Features")
    plt.ylabel("Mean Accuracy")
    plt.title("XGBoost Mean Accuracy vs. Number of Features")
    plt.savefig("XGBoost_Accuracy.png")
    plt.show()

# Define the range for number of features and other parameters
num_features_range = range(100, 800, 50)
n_folds = 5
output_csv = 'xgboost_all_fold_accuracies.csv'
plot_xgb_mean_accuracy_vs_features(X_selected, y, mi_scores_series, num_features_range, n_folds, output_csv)

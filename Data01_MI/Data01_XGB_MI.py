import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# Load the CSV data
df = pd.read_csv('Data01_MI/data1.csv', low_memory=False)
df_transposed = df.transpose()
df_transposed = df_transposed.drop(df_transposed.index[0])
df_transposed = df_transposed.drop_duplicates()

# Calculate percentage of missing values
null_values_percentage = round((df_transposed.isnull().sum() / df_transposed.shape[0]) * 100, 2)
print("Null Values Percentage:\n", null_values_percentage)

# Encode the target column (last column) using LabelEncoder
last_column = df_transposed.iloc[:, -1]  
encoder = LabelEncoder()
encoded_last_column = encoder.fit_transform(last_column)
df_transposed['Target'] = encoded_last_column
df_transposed = df_transposed.drop(df_transposed.columns[60676], axis=1)

# Handle non-numeric data types by converting object columns to categorical or numeric
object_cols = df_transposed.select_dtypes(include='object').columns
for col in object_cols:
    df_transposed[col] = LabelEncoder().fit_transform(df_transposed[col])

X = df_transposed.drop(columns=["Target"])
y = df_transposed["Target"]

# Calculate Mutual Information (MI) scores
mi_scores = mutual_info_classif(X, y)
mi_scores_series = pd.Series(mi_scores, index=X.columns, name="MI Scores")
mi_scores_df = pd.DataFrame(mi_scores_series)
mi_scores_df.to_csv("mi_scores_data1.csv")

plt.figure(figsize=(10, 6))
mi_scores_series.plot(kind="bar")
plt.title("Mutual Information Scores for Dataset Features")
plt.xlabel("Features")
plt.ylabel("MI Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("mi_scores_plot.png")
plt.show()

# Select features based on MI threshold
mi_threshold = 0
selected_features = mi_scores_series[mi_scores_series > mi_threshold].index.tolist()
df_selected = X[selected_features].copy()
df_selected['Target'] = y 
df_selected.to_csv("data1_preprocessed.csv", index=False)
print("Selected features and target saved to data1_preprocessed.csv")

# Function to train and evaluate XGBoost model
def train_and_evaluate_xgboost_with_varied_features(X, y, feature_counts):
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score

    mean_accuracies = []
    std_accuracies = []
    fold_results = []

    for num_features in feature_counts:
        X_selected_features = X.iloc[:, :num_features]
        
        # Initialize XGBoost Classifier
        xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, enable_categorical=True)
        
        # Perform cross-validation
        scores = cross_val_score(xgb_classifier, X_selected_features, y, cv=5, scoring='accuracy')

        for fold, accuracy in enumerate(scores, 1):
            fold_results.append({
                'Number of Features': num_features,
                'Fold': fold,
                'Accuracy': accuracy
            })

        mean_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)
        mean_accuracies.append(mean_accuracy)
        std_accuracies.append(std_accuracy)

        print(f"Number of Features: {num_features}")
        print(f"Mean Accuracy: {mean_accuracy:.2f}")
        print(f"Standard Deviation: {std_accuracy:.2f}")
        print("=" * 30)

    fold_results_df = pd.DataFrame(fold_results)
    fold_results_df.to_csv("XGBoost_accuracies.csv", index=False)
    print("Cross-validation fold results saved")

    return mean_accuracies, std_accuracies

feature_counts_to_test = list(range(100, 190, 20))
mean_accuracies, std_accuracies = train_and_evaluate_xgboost_with_varied_features(X[selected_features], y, feature_counts_to_test)

# Plot results
plt.figure(figsize=(10, 6))
plt.errorbar(feature_counts_to_test, mean_accuracies, yerr=std_accuracies, fmt='o-', capsize=4)
plt.title('XGBoost Accuracy vs. Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Mean Accuracy')
plt.grid(True)
plt.savefig("XGBoost_accuracy_vs_features.png")
plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv('Data01_FI/data1.csv')
df_transposed = df.transpose()
df_transposed = df_transposed.drop(df_transposed.index[0])

# Separate numeric and non-numeric columns
numeric_cols = df_transposed.select_dtypes(include=[np.number]).columns
non_numeric_cols = df_transposed.select_dtypes(exclude=[np.number]).columns

# Impute missing values for numeric columns with median
df_transposed[numeric_cols] = df_transposed[numeric_cols].fillna(df_transposed[numeric_cols].median())

# Impute missing values for non-numeric columns with mode
df_transposed[non_numeric_cols] = df_transposed[non_numeric_cols].fillna(df_transposed[non_numeric_cols].mode().iloc[0])

# Calculate and print null values percentage
null_values = round((df_transposed.isnull().sum() / df_transposed.shape[0]) * 100, 2)
print("Null values percentage:\n", null_values)

# Encode target variable
last_column = df_transposed.iloc[:, -1]
encoder = LabelEncoder()
encoded_last_column = encoder.fit_transform(last_column)
df_transposed['Target'] = encoded_last_column

# Drop the last column (which was originally the target)
df_transposed = df_transposed.drop(df_transposed.columns[-2], axis=1)

# Separate features and target
X = df_transposed.iloc[:, :-1]
y = df_transposed['Target']

# Training RandomForest for feature importance
print("Training RandomForest for feature importance...")
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X, y)
feature_importances = rf_classifier.feature_importances_

# Save feature importances to CSV
importance_series = pd.Series(feature_importances, index=X.columns, name="Feature Importances")
importance_series = importance_series.sort_values(ascending=False)
importance_series.to_csv('feature_importances.csv')

# Plot all feature importances
plt.figure(figsize=(20, 10))
importance_series.plot(kind='bar')
plt.title('Feature Importances from RandomForest')
plt.ylabel('Importance Score')
plt.xlabel('Features')
plt.xticks(rotation=90)
plt.savefig('all_feature_importances_plot.png')
plt.show()

# Prepare data for XGBoost
print("Preparing XGBoost model training...")
for column in X.columns:
    if X[column].dtype == 'object':
        X[column], _ = pd.factorize(X[column])

# One-hot encoding for any remaining categorical columns
X = pd.get_dummies(X, columns=X.select_dtypes(include='category').columns)

# Function to train and evaluate XGBoost
def train_and_evaluate_xgboost(X, y, num_features):
    top_features = importance_series.index[:num_features]
    X_top_features = X[top_features]
    xgb_classifier = XGBClassifier()
    scores = cross_val_score(xgb_classifier, X_top_features, y, cv=5)
    return scores, np.mean(scores), np.std(scores)

# Function to plot XGBoost mean accuracy vs. number of features
def plot_xgboost_mean_accuracy_vs_features(X, y, num_features_range):
    mean_accuracies = []
    std_devs = []
    results = []

    for num_features in num_features_range:
        print(f"Evaluating with {num_features} features...")
        fold_accuracies, mean_accuracy, std_dev = train_and_evaluate_xgboost(X, y, num_features)
        mean_accuracies.append(mean_accuracy)
        std_devs.append(std_dev)
        results.append({
            'Number of Features': num_features,
            'Mean Accuracy': mean_accuracy,
            'Standard Deviation': std_dev
        })

        print(f"Number of Features: {num_features}")
        for i, acc in enumerate(fold_accuracies):
            print(f"Fold {i+1} - Accuracy: {acc:.2f}")
        print(f"Mean Accuracy: {mean_accuracy:.2f}")
        print(f"Standard Deviation: {std_dev:.2f}")
        print()

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('xgboost_accuracy_vs_features.csv', index=False)

    # Plot accuracies with error bars
    plt.errorbar(num_features_range, mean_accuracies, yerr=std_devs, fmt='o-', label='Mean Accuracy')
    plt.xlabel("Number of Features")
    plt.ylabel("Mean Accuracy")
    plt.title("XGBoost Mean Accuracy vs. Number of Features with Standard Deviation")
    plt.legend()
    plt.savefig('xgboost_accuracy_vs_features.png')
    plt.show()

# Define range of features to evaluate
num_features_range = range(10, 50, 10)
plot_xgboost_mean_accuracy_vs_features(X, y, num_features_range)

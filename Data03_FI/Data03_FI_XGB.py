import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Data03_FI/top_1200_featuresN.csv')
print("Initial Dataframe:")
print(df.head())
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
missing_values = round((df.isnull().sum()/df.shape[0])*100, 2)
print("Percentage of missing values per column:")
print(missing_values)
last_column = df.iloc[:, -1]
encoder = LabelEncoder()
encoded_last_column = encoder.fit_transform(last_column)
df['Target'] = encoded_last_column
df = df.drop(df.columns[1200], axis=1)
print("DataFrame after encoding the target:")
print(df.head())

X = df.drop(columns=["Target"])
y = df["Target"]

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X, y)
feature_importances = rf_classifier.feature_importances_
importance_series = pd.Series(feature_importances, index=X.columns, name="Feature Importances")
importance_series = importance_series.sort_values(ascending=False)
print("Feature importances:")
print(importance_series)

importance_threshold = 0.0
selected_features = importance_series[importance_series > importance_threshold].index.tolist()
X_selected = X[selected_features]
print("Selected features based on feature importance:")
print(X_selected.head())

def train_and_evaluate_xgboost(X, y, num_features):
    top_features = X_selected.columns[:num_features]
    X_top_features = X_selected[top_features]
    xgb_classifier = XGBClassifier()
    scores = cross_val_score(xgb_classifier, X_top_features, y, cv=5)

    return scores

# Plot XGBoost mean accuracy vs. number of features
def plot_xgboost_mean_accuracy_vs_features(X, y, num_features_range):
    mean_accuracies = []
    std_accuracies = []
    results = []

    for num_features in num_features_range:
        fold_accuracies = train_and_evaluate_xgboost(X, y, num_features)
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        mean_accuracies.append(mean_accuracy)
        std_accuracies.append(std_accuracy)

        results.append([num_features, mean_accuracy, std_accuracy])

        for i, acc in enumerate(fold_accuracies):
            print(f"Fold {i+1} - Accuracy: {acc:.2f}")

        print(f"Mean Accuracy ({num_features} features): {mean_accuracy:.2f}")
        print(f"Standard Deviation ({num_features} features): {std_accuracy:.2f}")
        print()

    results_df = pd.DataFrame(results, columns=['Number of Features', 'Mean Accuracy', 'Std Deviation'])
    results_df.to_csv('xgboost_accuracy_results.csv', index=False)

    plt.errorbar(num_features_range, mean_accuracies, yerr=std_accuracies, fmt='o-', capsize=4)
    plt.xlabel("Number of Features")
    plt.ylabel("Mean Accuracy")
    plt.title("XGBoost Mean Accuracy vs. Number of Features")
    plt.tight_layout()
    plt.savefig('xgboost_mean_accuracy.png')
    plt.show()

num_features_range = range(100, 800, 100)
plot_xgboost_mean_accuracy_vs_features(X_selected, y, num_features_range)

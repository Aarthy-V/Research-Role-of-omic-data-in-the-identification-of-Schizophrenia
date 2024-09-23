import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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
X_selected.to_csv('selected_features.csv', index=False)

# Define a function to train and evaluate Random Forest with different feature counts
def train_and_evaluate_random_forest_with_varied_features(X, y, feature_counts):
    mean_accuracies = []
    std_accuracies = []
    n_estimators = 50  

    results = [] 

    for num_features in feature_counts:
        X_selected_features = X.iloc[:, :num_features]
        rf_classifier = RandomForestClassifier(class_weight='balanced', n_estimators=n_estimators, random_state=42)
        scores = cross_val_score(rf_classifier, X_selected_features, y, cv=5, scoring='accuracy')

        mean_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)
        mean_accuracies.append(mean_accuracy)
        std_accuracies.append(std_accuracy)

        results.append([num_features, mean_accuracy, std_accuracy])

        print(f"Number of Features: {num_features}")
        for fold, accuracy in enumerate(scores, 1):
            print(f"Fold {fold} - Accuracy: {accuracy:.2f}")

        print(f"Mean Accuracy: {mean_accuracy:.2f}")
        print(f"Standard Deviation: {std_accuracy:.2f}")
        print("=" * 30)

    results_df = pd.DataFrame(results, columns=["Number of Features", "Mean Accuracy", "Standard Deviation"])
    results_df.to_csv('rf_feature_selection_results.csv', index=False)

    return mean_accuracies, std_accuracies

feature_counts_to_test = list(range(100, 800, 100))
mean_accuracies, std_accuracies = train_and_evaluate_random_forest_with_varied_features(X_selected, y, feature_counts_to_test)

plt.figure(figsize=(10, 6))
plt.errorbar(feature_counts_to_test, mean_accuracies, yerr=std_accuracies, fmt='o-', capsize=4)
plt.title('Random Forest Accuracy vs. Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Mean Accuracy')
plt.grid(True)
plt.savefig('rf_accuracy_vs_features.png') 
plt.show()



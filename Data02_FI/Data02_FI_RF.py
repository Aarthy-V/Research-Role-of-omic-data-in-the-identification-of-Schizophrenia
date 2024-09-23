import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from collections import Counter

# Load the dataset
df = pd.read_csv('Data02_FI/data2.csv')
df_transposed = df.transpose()
df_transposed = df_transposed.drop(df_transposed.index[0])
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
importance_series.to_csv('feature_importances.csv')

plt.figure(figsize=(10, 6))
importance_series.plot(kind='bar')
plt.title('Feature Importances from RandomForest')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importances_plot.png') 
plt.show()

importance_threshold = 0.0  
selected_features = importance_series[importance_series > importance_threshold].index.tolist()
X_selected = X_resampled[selected_features]
print("Final X_selected shape:", X_selected.shape)
print("Final y_resampled shape:", y_resampled.shape)

# Define a function to train and evaluate Random Forest with different feature counts
def train_and_evaluate_random_forest_with_varied_features(X, y, feature_counts):
    mean_accuracies = []
    std_accuracies = []
    n_estimators = 50 

    for num_features in feature_counts:
        X_selected_features = X.iloc[:, :num_features]
        rf_classifier = RandomForestClassifier(class_weight='balanced', n_estimators=n_estimators, random_state=42)
        scores = cross_val_score(rf_classifier, X_selected_features, y, cv=5, scoring='accuracy')

        mean_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)
        mean_accuracies.append(mean_accuracy)
        std_accuracies.append(std_accuracy)

        print(f"Number of Features: {num_features}")
        for fold, accuracy in enumerate(scores, 1):
            print(f"Fold {fold} - Accuracy: {accuracy:.2f}")

        print(f"Mean Accuracy: {mean_accuracy:.2f}")
        print(f"Standard Deviation: {std_accuracy:.2f}")
        print("=" * 30)

    return mean_accuracies, std_accuracies

feature_counts_to_test = list(range(100, 150, 10))
mean_accuracies, std_accuracies = train_and_evaluate_random_forest_with_varied_features(X_selected, y_resampled, feature_counts_to_test)

plt.figure(figsize=(10, 6))
plt.errorbar(feature_counts_to_test, mean_accuracies, yerr=std_accuracies, fmt='o-', capsize=4)
plt.title('Random Forest Accuracy vs. Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Mean Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_vs_features.png')  
plt.show()

accuracy_data = pd.DataFrame({
    'Number of Features': feature_counts_to_test,
    'Mean Accuracy': mean_accuracies,
    'Standard Deviation': std_accuracies
})
accuracy_data.to_csv('accuracy_vs_features.csv', index=False)



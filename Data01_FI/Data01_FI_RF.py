import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv('Data01_FI/data1.csv')
df_transposed = df.transpose()
df_transposed = df_transposed.drop(df_transposed.index[0])
last_column = df_transposed.iloc[:, -1] 
encoder = LabelEncoder()
encoded_last_column = encoder.fit_transform(last_column)
df_transposed['Target'] = encoded_last_column
df_transposed = df_transposed.drop(df_transposed.columns[60676], axis=1)
X = df_transposed.iloc[:, :-1]  
y = df_transposed['Target'] 

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X, y)
feature_importances = rf_classifier.feature_importances_
importance_series = pd.Series(feature_importances, index=X.columns, name="Feature Importances")
importance_series.to_csv('feature_importances.csv')

plt.figure(figsize=(20, 10))
importance_series.plot(kind='bar')
plt.title('Feature Importances')
plt.ylabel('Importance Score')
plt.xlabel('Features')
plt.xticks(rotation=90) 
plt.savefig('all_feature_importances_plot.png')
plt.show()

importance_threshold = 0.0
selected_features = importance_series[importance_series > importance_threshold].index.tolist()
X_selected = X[selected_features]

# Define a function to train and evaluate Random Forest with different feature counts
def train_and_evaluate_random_forest_with_varied_features(X, y, feature_counts):
    mean_accuracies = []
    std_accuracies = []
    fold_accuracies = []
    n_estimators = 50 

    for num_features in feature_counts:
        X_selected_features = X.iloc[:, :num_features]
        rf_classifier = RandomForestClassifier(class_weight='balanced', n_estimators=n_estimators, random_state=42)
        scores = cross_val_score(rf_classifier, X_selected_features, y, cv=5, scoring='accuracy')
        mean_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)
        mean_accuracies.append(mean_accuracy)
        std_accuracies.append(std_accuracy)
        fold_accuracies.append(scores)

        print(f"Number of Features: {num_features}")
        for fold, accuracy in enumerate(scores, 1):
            print(f"Fold {fold} - Accuracy: {accuracy:.2f}")

        print(f"Mean Accuracy: {mean_accuracy:.2f}")
        print(f"Standard Deviation: {std_accuracy:.2f}")
        print("=" * 30)

    return mean_accuracies, std_accuracies, fold_accuracies

feature_counts_to_test = list(range(100, 190, 10))
mean_accuracies, std_accuracies, fold_accuracies = train_and_evaluate_random_forest_with_varied_features(X_selected, y, feature_counts_to_test)
fold_accuracies_df = pd.DataFrame(fold_accuracies, columns=[f'Fold_{i+1}' for i in range(10)], index=feature_counts_to_test)
fold_accuracies_df.to_csv('RF_fold_accuracies.csv')

plt.figure(figsize=(10, 6))
plt.errorbar(feature_counts_to_test, mean_accuracies, yerr=std_accuracies, fmt='o-', capsize=4)
plt.title('Random Forest Accuracy vs. Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Mean Accuracy')
plt.grid(True)
plt.savefig('random_forest_accuracy_vs_features.png')
plt.show()


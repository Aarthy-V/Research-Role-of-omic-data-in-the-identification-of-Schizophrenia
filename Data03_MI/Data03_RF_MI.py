import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('top_1200_featuresN.csv')
df.duplicated()
round((df.isnull().sum() / df.shape[0]) * 100, 2)
last_column = df.iloc[:, -1]
encoder = LabelEncoder()
encoded_last_column = encoder.fit_transform(last_column)
df['Target'] = encoded_last_column
df = df.drop(df.columns[1200], axis=1)

X = df.drop(columns=["Target"])
y = df["Target"]

mi_scores = mutual_info_classif(X, y)
mi_scores_series = pd.Series(mi_scores, index=X.columns, name="MI Scores")
mi_scores_series = mi_scores_series.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
plt.scatter(range(len(mi_scores_series)), mi_scores_series, s=50, alpha=0.5)
plt.title("Mutual Information Scores for Dataset Features")
plt.xlabel("Feature Index")
plt.ylabel("MI Score")
plt.grid(True)
plt.tight_layout()
plt.savefig('mi_score_graph.png')  
plt.show()


mi_scores_df = pd.DataFrame(mi_scores_series)
mi_scores_df.to_csv("mi_scores_1200.csv", index=True)


def train_and_evaluate_random_forest_with_varied_features(X, y, feature_counts):
    mean_accuracies = []
    std_accuracies = []

    n_estimators = 50  

    for num_features in feature_counts:
        X_selected_features = X.iloc[:, :num_features]
        rf_classifier = RandomForestClassifier(class_weight='balanced', n_estimators=n_estimators, random_state=42)
        scores = cross_val_score(rf_classifier, X_selected_features, y, cv=5)

        mean_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)
        mean_accuracies.append(mean_accuracy)
        std_accuracies.append(std_accuracy)

    return mean_accuracies, std_accuracies

feature_counts_to_test = list(range(100, 800, 100))
mean_accuracies, std_accuracies = train_and_evaluate_random_forest_with_varied_features(X, y, feature_counts_to_test)

best_index = np.argmax(mean_accuracies)
best_num_features = feature_counts_to_test[best_index]
best_mean_accuracy = mean_accuracies[best_index]
best_std_accuracy = std_accuracies[best_index]

plt.figure(figsize=(10, 6))
plt.errorbar(feature_counts_to_test, mean_accuracies, yerr=std_accuracies, fmt='o-', capsize=4)
plt.title('Random Forest Accuracy vs. Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Mean Accuracy')
plt.grid(True)
plt.savefig('rf_accuracy_graph.png') 
plt.show()

accuracy_results = pd.DataFrame({
    'Number of Features': feature_counts_to_test,
    'Mean Accuracy': mean_accuracies,
    'Standard Deviation': std_accuracies
})
accuracy_results.to_csv('rf_accuracy_results.csv', index=False)



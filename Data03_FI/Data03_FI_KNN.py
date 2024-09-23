import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('Data03_FI/top_1200_featuresN.csv')
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
missing_values = round((df.isnull().sum() / df.shape[0]) * 100, 2)
print("Percentage of missing values per column:")
print(missing_values)
last_column = df.iloc[:, -1]
encoder = LabelEncoder()
encoded_last_column = encoder.fit_transform(last_column)
df['Target'] = encoded_last_column
df = df.drop(df.columns[1200], axis=1)

X = df.drop(columns=["Target"])
y = df["Target"]
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X, y)
feature_importances = rf_classifier.feature_importances_
importance_series = pd.Series(feature_importances, index=X.columns, name="Feature Importances")
importance_series = importance_series.sort_values(ascending=False)

importance_threshold = 0.0
selected_features = importance_series[importance_series > importance_threshold].index.tolist()
X_selected = X[selected_features]

def train_and_evaluate_knn_with_varied_features(X, y, feature_counts, k_values):
    results = []

    for num_features in feature_counts:
        top_features = X_selected.columns[:num_features]
        X_top_features = X_selected[top_features]

        for k in k_values:
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn_classifier, X_top_features, y, cv=5)
            mean_accuracy = scores.mean()

            results.append((num_features, k, mean_accuracy))

    return results

feature_counts_to_test = list(range(100, 800, 100))
k_values_to_test = list(range(3, 21))
results = train_and_evaluate_knn_with_varied_features(X, y, feature_counts_to_test, k_values_to_test)
result_df = pd.DataFrame(results, columns=["Num Features", "k Value", "Mean Accuracy"])
result_df.to_csv('knn_results.csv', index=False)

heatmap_data = result_df.pivot(index="k Value", columns="Num Features", values="Mean Accuracy")

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("KNN Mean Accuracy Heatmap")
plt.xlabel("Number of Features")
plt.ylabel("k Value")
plt.savefig('knn_heatmap.png')
plt.show()



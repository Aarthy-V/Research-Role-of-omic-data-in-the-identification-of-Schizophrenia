import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('top_1200_featuresN.csv')
df.drop_duplicates(inplace=True)
df = df.dropna() 
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
mi_scores_df = pd.DataFrame(mi_scores_series)
mi_scores_df.to_csv("mi_scores_1200.csv")

plt.figure(figsize=(10, 6))
plt.scatter(range(len(mi_scores_series)), mi_scores_series, s=50, alpha=0.5)
plt.title("Mutual Information Scores for Dataset Features")
plt.xlabel("Feature Index")
plt.ylabel("MI Score")
plt.grid(True)
plt.tight_layout()
plt.savefig('mi_scores_graph.png')
plt.show()

# Define a function to train and evaluate KNN with different feature counts and k values
def train_and_evaluate_knn_with_varied_features(X, y, feature_counts, k_values):
    results = []

    for num_features in feature_counts:
        top_features = mi_scores_series.nlargest(num_features).index.tolist()
        X_top_features = X[top_features]

        for k in k_values:
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn_classifier, X_top_features, y, cv=10)
            mean_accuracy = scores.mean()
            results.append((num_features, k, mean_accuracy))

    return results

feature_counts_to_test = list(range(100, 800, 100))
k_values_to_test = list(range(3, 21))
results = train_and_evaluate_knn_with_varied_features(X, y, feature_counts_to_test, k_values_to_test)
result_df = pd.DataFrame(results, columns=["Num Features", "k Value", "Mean Accuracy"])
result_df.to_csv("knn_results.csv", index=False)
heatmap_data = result_df.pivot(index="k Value", columns="Num Features", values="Mean Accuracy")

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("KNN Mean Accuracy Heatmap")
plt.xlabel("Number of Features")
plt.ylabel("k Value")
plt.tight_layout()
plt.savefig('knn_accuracy_heatmap.png')
plt.show()





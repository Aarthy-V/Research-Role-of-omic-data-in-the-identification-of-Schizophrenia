import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

# Loads a CSV file (data1.csv) into a pandas DataFrame and transposes it 
# Removes the first row and any duplicate rows, then 
# calculates the percentage of missing (null) values in each column. The last column is encoded 
# using LabelEncoder to transform categorical values into numerical ones, and encoded column 
# is stored as a new 'Target' column. The code then drops the 60676th column  
# and prepares the feature set (X) by excluding the 'Target' column, which is stored separately as the target variable (y).
df = pd.read_csv('Data01/data1.csv', low_memory=False)
df_transposed = df.transpose()
df_transposed = df_transposed.drop(df_transposed.index[0])
df_transposed = df_transposed.drop_duplicates()
null_values_percentage = round((df_transposed.isnull().sum() / df_transposed.shape[0]) * 100, 2)
print("Null Values Percentage:\n", null_values_percentage)
last_column = df_transposed.iloc[:, -1]
encoder = LabelEncoder()
encoded_last_column = encoder.fit_transform(last_column)
df_transposed['Target'] = encoded_last_column
df_transposed = df_transposed.drop(df_transposed.columns[-2], axis=1)
X_preprocessed = df_transposed.drop(columns=["Target"])
y_preprocessed = df_transposed["Target"]

# Calculates the Mutual Information (MI) scores between features (X) and the target variable (y) 
# The MI scores are stored in a pandas Series, with the feature names as the index, 
# and then converted to a DataFrame. The scores are saved to a CSV file. 
# Next, a bar plot is generated to visualize the MI scores for the dataset's features. 
# And saved as an image file 
mi_scores = mutual_info_classif(X_preprocessed, y_preprocessed)
mi_scores_series = pd.Series(mi_scores, index=X_preprocessed.columns, name="MI Scores")
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

mi_scores_series = pd.read_csv('mi_scores_data1.csv', index_col=0)['MI Scores']

# Define a function to train and evaluate KNN with varied feature counts and k values
def train_and_evaluate_knn_with_varied_features(X, y, mi_scores_series, feature_counts, k_values):
    results = []
    mi_scores_series.index = X.columns
    
    for num_features in feature_counts:
        top_features = mi_scores_series.nlargest(num_features).index.tolist()  # Get top feature names
        X_top_features = X[top_features] 
        
        for k in k_values:
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn_classifier, X_top_features, y, cv=5)
            mean_accuracy = scores.mean()
            results.append((num_features, k, mean_accuracy))

    return results

feature_counts_to_test = list(range(50, 160, 10))
k_values_to_test = list(range(3, 21))


results = train_and_evaluate_knn_with_varied_features(X_preprocessed, y_preprocessed, mi_scores_series, feature_counts_to_test, k_values_to_test)

result_df = pd.DataFrame(results, columns=["Num Features", "k Value", "Mean Accuracy"])
heatmap_data = result_df.pivot(index="k Value", columns="Num Features", values="Mean Accuracy")


plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("KNN Mean Accuracy Heatmap")
plt.xlabel("Number of Features")
plt.ylabel("k Value")
plt.savefig("knn_accuracy_heatmap.png")
plt.show()


result_df.to_csv("knn_accuracies.csv", index=False)



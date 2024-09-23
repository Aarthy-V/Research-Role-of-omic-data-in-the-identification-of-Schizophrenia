import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

# Loads a CSV file (data1.csv) into a pandas DataFrame and transposes it 
# Removes the first row and any duplicate rows, then 
# calculates the percentage of missing (null) values in each column. The last column is encoded 
# using LabelEncoder to transform categorical values into numerical ones, and encoded column 
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
df_transposed = df_transposed.drop(df_transposed.columns[60676], axis=1)

# Define features X and target y
X = df_transposed.drop(columns=["Target"])
y = df_transposed["Target"]

# Calculate mutual information scores
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


mi_threshold = 0  
selected_features = mi_scores_series[mi_scores_series > mi_threshold].index.tolist()
df_selected = X[selected_features].copy()
df_selected['Target'] = y 
df_selected.to_csv("data1_preprocessed.csv", index=False)
print("Selected features and target saved to data1_preprocessed.csv")

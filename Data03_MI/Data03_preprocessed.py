
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

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
plt.savefig('Mi score graph.png')
plt.show()


mi_scores_df = pd.DataFrame(mi_scores_series)
mi_scores_df.to_csv("mi_scores_1200.csv")
mi_threshold = 0
selected_features = mi_scores_series[mi_scores_series > mi_threshold].index.tolist()
X_selected = X[selected_features]
print(X_selected)

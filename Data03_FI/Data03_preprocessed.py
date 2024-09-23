import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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
X_selected.to_csv('selected_features.csv', index=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=importance_series[:20], y=importance_series.index[:20], palette="viridis")
plt.title('Top 20 Feature Importances')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig("Feature_Importance_Graph.png")
plt.show()
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


df = pd.read_csv('Data01_FI/data1.csv')
df_transposed = df.transpose()
df_transposed = df_transposed.drop(df_transposed.index[0])
null_values = round((df_transposed.isnull().sum()/df_transposed.shape[0])*100, 2)
print("Null values percentage:\n", null_values)
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
importance_series = importance_series.sort_values(ascending=False)
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
X_selected.to_csv('selected_features.csv', index=False)

print("Selected Features:\n", X_selected)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from collections import Counter


df = pd.read_csv('Data02_FI/data2.csv')
df_transposed = df.transpose()
df_transposed = df_transposed.drop(df_transposed.index[0])
duplicates = df_transposed.duplicated()
print("Number of duplicate rows:", duplicates.sum())
missing_percentage = round((df_transposed.isnull().sum() / df_transposed.shape[0]) * 100, 2)
print("Percentage of missing values per column:\n", missing_percentage)
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

resampled_df = pd.DataFrame({'target': y_resampled})
class_counts = resampled_df['target'].value_counts()
print("Control Count:", class_counts[0])
print("Patient Count:", class_counts[1])

print("Resampled X shape:", X_resampled.shape)
print("Resampled y shape:", y_resampled.shape)

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_resampled, y_resampled)

feature_importances = rf_classifier.feature_importances_
importance_series = pd.Series(feature_importances, index=X_resampled.columns, name="Feature Importances")
importance_series = importance_series.sort_values(ascending=False)

print("Feature importances:\n", importance_series)

importance_threshold = 0.0 
selected_features = importance_series[importance_series > importance_threshold].index.tolist()
X_selected = X_resampled[selected_features]

print("Selected features:\n", X_selected.columns)
print("Final X_selected shape:", X_selected.shape)
print("Final y_resampled shape:", y_resampled.shape)

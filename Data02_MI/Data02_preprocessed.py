import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

def preprocess_data(file_path, target_sample_count=80, random_state=42, output_features_path='X_resampled.csv', output_target_path='y_resampled.csv'):

    # Load and preprocess data
    df = pd.read_csv(file_path)
    df_transposed = df.transpose()
    df_transposed = df_transposed.drop(df_transposed.index[0])
    last_column = df_transposed.iloc[:, -1]
    encoder = LabelEncoder()
    encoded_last_column = encoder.fit_transform(last_column)
    df_transposed['Target'] = encoded_last_column
    df_transposed = df_transposed.drop(df_transposed.columns[-2], axis=1)
    X = df_transposed.drop(columns=["Target"])
    y = df_transposed["Target"]
    
    # Resample the data
    ros = RandomOverSampler(sampling_strategy={0: target_sample_count, 1: target_sample_count}, random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    resampled_df = pd.DataFrame({'target': y_resampled})
    class_counts_resampled = resampled_df['target'].value_counts()
    
    # Print resampled class counts and shapes
    print("\nResampled Control Count:", class_counts_resampled[0])
    print("Resampled Patient Count:", class_counts_resampled[1])
    print("\nResampled X shape:", X_resampled.shape)
    print("Resampled y shape:", y_resampled.shape)
    
    # Save the resampled data to CSV files
    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    X_resampled_df.to_csv(output_features_path, index=False)
    y_resampled_df = pd.DataFrame(y_resampled, columns=['Target'])
    y_resampled_df.to_csv(output_target_path, index=False)
    
    return X_resampled, y_resampled

if __name__ == "__main__":
    file_path = 'Data02/data2.csv' 
    X_resampled, y_resampled = preprocess_data(file_path)

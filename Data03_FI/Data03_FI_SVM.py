import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('Data03_FI/top_1200_featuresN.csv')
print("Initial Dataframe:")
print(df.head())

# Check for duplicates and missing values
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
missing_values = round((df.isnull().sum()/df.shape[0])*100, 2)
print("Percentage of missing values per column:")
print(missing_values)

# Encode the target variable
last_column = df.iloc[:, -1] 
encoder = LabelEncoder()
encoded_last_column = encoder.fit_transform(last_column)
df['Target'] = encoded_last_column
df = df.drop(df.columns[1200], axis=1)
print("DataFrame after encoding the target:")
print(df.head())

# Split features and target
X = df.drop(columns=["Target"])
y = df["Target"]
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Feature selection using RandomForestClassifier with random_state
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X, y)
feature_importances = rf_classifier.feature_importances_
importance_series = pd.Series(feature_importances, index=X.columns, name="Feature Importances")
importance_series = importance_series.sort_values(ascending=False)

print("Feature importances:")
print(importance_series)

# Select features based on importance
importance_threshold = 0.0
selected_features = importance_series[importance_series > importance_threshold].index.tolist()
X_selected = X[selected_features]

# Function to train and evaluate SVM with StratifiedKFold for random state control
def train_and_evaluate_svm(X, y, num_features, n_folds, random_state=42):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=num_features, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create SVM classifier
    svm_classifier = SVC(kernel='linear', decision_function_shape='ovr')
    
    # Use StratifiedKFold for cross-validation with a fixed random_state
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Perform cross-validation
    scores = cross_val_score(svm_classifier, X_pca, y, cv=cv)

    for i, acc in enumerate(scores):
        print(f"Fold {i + 1} - Accuracy: {acc:.2f}")

    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)
    print(f"Mean Accuracy ({num_features} features): {mean_accuracy:.2f}")
    print(f"Standard Deviation ({num_features} features): {std_accuracy:.2f}")
    print()

    return mean_accuracy, std_accuracy

# Function to plot SVM mean accuracy vs. number of features and save results
def plot_svm_mean_accuracy_vs_features(X, y, num_features_range, n_folds, results_csv_path, plot_png_path, random_state=42):
    mean_accuracies = []
    std_accuracies = []
    feature_numbers = []

    for num_features in num_features_range:
        mean_accuracy, std_accuracy = train_and_evaluate_svm(X, y, num_features, n_folds, random_state)
        mean_accuracies.append(mean_accuracy)
        std_accuracies.append(std_accuracy)
        feature_numbers.append(num_features)

    results_df = pd.DataFrame({
        'Number of Features': feature_numbers,
        'Mean Accuracy': mean_accuracies,
        'Standard Deviation': std_accuracies
    })
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

    plt.errorbar(num_features_range, mean_accuracies, yerr=std_accuracies, fmt='o-', capsize=4)
    plt.xlabel("Number of Features")
    plt.ylabel("Mean Accuracy")
    plt.title("SVM Mean Accuracy (Linear Kernel) vs. Number of Features")
    plt.savefig(plot_png_path)
    print(f"Plot saved to {plot_png_path}")
    plt.show()

# Define range of features and number of folds for cross-validation
num_features_range = range(100, 800, 50) 
n_folds = 5 
results_csv_path = 'svm_results.csv'
plot_png_path = 'svm_accuracy_plot.png'

# Plot SVM mean accuracy vs. number of features
plot_svm_mean_accuracy_vs_features(X_selected, y, num_features_range, n_folds, results_csv_path, plot_png_path)

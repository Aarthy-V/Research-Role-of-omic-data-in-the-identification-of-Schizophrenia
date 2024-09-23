# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

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
df_transposed = df_transposed.drop(df_transposed.columns[60676], axis=1)
X = df_transposed.drop(columns=["Target"])
y = df_transposed["Target"]

# Calculates the Mutual Information (MI) scores between features (X) and the target variable (y) 
# The MI scores are stored in a pandas Series, with the feature names as the index, 
# and then converted to a DataFrame. The scores are saved to a CSV file. 
# Next, a bar plot is generated to visualize the MI scores for the dataset's features. 
# And saved as an image file 
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

# Filters features based on a specified Mutual Information (MI) score greater than the 0. 
# Selects the feature names from mi_scores_series that have an MI score greater than the threshold and 
# creates a new DataFrame containing only those features. The target variable (y) is then added 
# to the DataFrame. Finally, the selected features and target are saved to a CSV file. 
mi_threshold = 0
selected_features = mi_scores_series[mi_scores_series > mi_threshold].index.tolist()
df_selected = X[selected_features].copy()
df_selected['Target'] = y
df_selected.to_csv("data1_preprocessed.csv", index=False)
print("Selected features and target saved to data1_preprocessed.csv")

# Reads a preprocessed CSV file (data1_preprocessed.csv) into a pandas DataFrame (df). 
# Separates the DataFrame into features (X) and the target variable (y). The target column, 
# named "Target", is dropped from df to create X, which contains only the features, while y is assigned 
# the values of the "Target" column.
df = pd.read_csv('data1_preprocessed.csv')
X = df.drop(columns=["Target"])
y = df["Target"]


# This function, trains and evaluates a Support Vector Machine (SVM) classifier using the 
# top num_features from the feature set X and target variable y. First, it selects the top features from X based 
# on the specified number. The SVM classifier is initialized with a linear kernel and a one-vs-rest (OVR) 
# It then performs 5-fold cross-validation using cross_val_score, which 
# splits the data into 5 folds, trains the model on each fold, and evaluates its performance. 
def train_and_evaluate_svm(X, y, num_features, random_state=42):
    top_features = X.columns[:num_features].tolist()
    X_top_features = X[top_features]
    svm_classifier = SVC(kernel='linear', random_state=random_state)
    scores = cross_val_score(svm_classifier, X_top_features, y, cv=5)
    return scores


def plot_svm_mean_accuracy_vs_features(X, y, num_features_range):
    mean_accuracies = []
    std_accuracies = []
    # For each value in num_features_range, the function calls train_and_evaluate_svm 
    # to get the cross-validation accuracy for the top num_features. 
    # It then calculates and stores the mean and standard deviation of these accuracies.
    for num_features in num_features_range:
        fold_accuracies = train_and_evaluate_svm(X, y, num_features, random_state=42)
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        mean_accuracies.append(mean_accuracy)
        std_accuracies.append(std_accuracy)

        # For each feature count, it prints the accuracy of each fold, as well as the overall mean accuracy and standard deviation.
        for i, acc in enumerate(fold_accuracies):
            print(f"Fold {i+1} - Accuracy: {acc:.2f}")

        print(f"Mean Accuracy ({num_features} features): {mean_accuracy:.2f}")
        print(f"Standard Deviation ({num_features} features): {std_accuracy:.2f}")
        print()

    plt.errorbar(num_features_range, mean_accuracies, yerr=std_accuracies, fmt='o-', capsize=4)
    plt.xlabel("Number of Features")
    plt.ylabel("Mean Accuracy")
    plt.title("SVM Mean Accuracy (Linear Kernel) vs. Number of Features")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("svm_accuracy_vs_features.png")
    plt.show()

    # The accuracy statistics for each feature count are saved to a CSV file named svm_accuracies.csv.
    accuracy_df = pd.DataFrame({
        "Number of Features": num_features_range,
        "Mean Accuracy": mean_accuracies,
        "Std Deviation": std_accuracies
    })
    accuracy_df.to_csv("svm_accuracies.csv", index=False)

    print("Accuracy results saved to svm_accuracies.csv")
    print("Plot saved as svm_accuracy_vs_features.png")


num_features_range = list(range(100, 190, 20))
plot_svm_mean_accuracy_vs_features(X, y, num_features_range)

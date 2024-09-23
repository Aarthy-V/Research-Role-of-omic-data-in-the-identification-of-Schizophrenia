import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
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

# Function to train and evaluate Random Forest model
def train_and_evaluate_random_forest_with_varied_features(X, y, feature_counts):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    # For each value in num_features_range, the function calls train_and_evaluate_svm 
    # to get the cross-validation accuracy for the top num_features. 
    # It then calculates and stores the mean and standard deviation of these accuracies.
    mean_accuracies = []
    std_accuracies = []

    fold_results = []

    # A Random Forest classifier (rf_classifier) with 50 trees (n_estimators=50) and balanced class weights is trained using 5-fold cross-validation.
    n_estimators = 50  

    for num_features in feature_counts:
        X_selected_features = X.iloc[:, :num_features]
        rf_classifier = RandomForestClassifier(class_weight='balanced', n_estimators=n_estimators, random_state=42)
        scores = cross_val_score(rf_classifier, X_selected_features, y, cv=5, scoring='accuracy')

        for fold, accuracy in enumerate(scores, 1):
            fold_results.append({
                'Number of Features': num_features,
                'Fold': fold,
                'Accuracy': accuracy
            })

        mean_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)
        mean_accuracies.append(mean_accuracy)
        std_accuracies.append(std_accuracy)

        print(f"Number of Features: {num_features}")
        print(f"Mean Accuracy: {mean_accuracy:.2f}")
        print(f"Standard Deviation: {std_accuracy:.2f}")
        print("=" * 30)

    fold_results_df = pd.DataFrame(fold_results)
    fold_results_df.to_csv("RF_accuracies.csv", index=False)
    print("Cross-validation fold results saved")

    return mean_accuracies, std_accuracies


feature_counts_to_test = list(range(100, 190, 20))
mean_accuracies, std_accuracies = train_and_evaluate_random_forest_with_varied_features(X[selected_features], y, feature_counts_to_test)

plt.figure(figsize=(10, 6))
plt.errorbar(feature_counts_to_test, mean_accuracies, yerr=std_accuracies, fmt='o-', capsize=4)
plt.title('Random Forest Accuracy vs. Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Mean Accuracy')
plt.grid(True)
plt.savefig("RF_accuracy_vs_features.png")
plt.show()

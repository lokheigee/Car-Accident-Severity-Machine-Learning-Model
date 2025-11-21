import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("categorical_df.csv")

# Drop ID columns
df = df.drop(columns=["Unnamed: 0", "VEHICLE_ID"], errors='ignore')

# Re-map LIGHT_CONDITION into 3 categories: 0 = Low, 1 = Medium, 2 = High
def simplify_condition(val):
    if val <= 3:
        return 0 # Low
    elif val <= 6:
        return 1 # Medium
    else:
        return 2 # High

target_col = 'LIGHT_CONDITION'
y = df[target_col].apply(simplify_condition)
X = df.drop(columns=[target_col])

# Handle missing values & encode categoricals
for col in X.columns:
    if X[col].dtype in ['float64', 'int64']:
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna(X[col].mode()[0])
        X[col] = LabelEncoder().fit_transform(X[col])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model 1 - Decision Tree Classifier
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)

# Model 2 - K-Nearest Neighbors Classifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)

# Print classification reports
print("=== Decision Tree Classifier Report ===")
print(classification_report(y_test, tree_preds))
print("\n=== K-Nearest Neighbors Classifier Report ===")
print(classification_report(y_test, knn_preds))

# Confusion matrices
tree_cm = confusion_matrix(y_test, tree_preds)
knn_cm = confusion_matrix(y_test, knn_preds)

# Plot Decision Tree confusion matrix
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix=tree_cm, display_labels=["Low", "Medium", "High"]).plot(cmap=plt.cm.Blues)
plt.title("Decision Tree - Confusion Matrix")
plt.show()

# Plot KNN confusion matrix
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix=knn_cm, display_labels=["Low", "Medium", "High"]).plot(cmap=plt.cm.Oranges)
plt.title("K-Nearest Neighbors - Confusion Matrix")
plt.show()

# F1 scores
tree_f1_macro = f1_score(y_test, tree_preds, average='macro')
tree_f1_weighted = f1_score(y_test, tree_preds, average='weighted')
knn_f1_macro = f1_score(y_test, knn_preds, average='macro')
knn_f1_weighted = f1_score(y_test, knn_preds, average='weighted')

(tree_f1_macro, tree_f1_weighted, knn_f1_macro, knn_f1_weighted)

# # Plot the decision tree
# plt.figure(figsize=(20, 10))
# plot_tree(tree_model,
#           feature_names=X.columns,
#           class_names=["Low", "Medium", "High"],
#           filled=True)
# plt.title("Decision Tree Classifier (Grouped LIGHT_CONDITION)")
# plt.show()

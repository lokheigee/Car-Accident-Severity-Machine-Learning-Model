import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# Load the dataset
df = pd.read_csv('categorical_df.csv')

# Create severity categories based on ASI (Accident Severity Index)
def create_severity_category(asi):
    """
    Categorize accidents into three severity levels:
    - Low: ASI <= 0.33
    - Medium: 0.33 < ASI <= 0.66
    - High: ASI > 0.66
    """
    if asi <= 0.33:
        return 0  # Low
    elif asi <= 0.66:
        return 1  # Medium
    else:
        return 2  # High

# Apply severity categorization
df['severity'] = df['ASI'].apply(create_severity_category)

# Select features for modeling
feature_columns = ['VEHICLE_BODY_STYLE', 'VEHICLE_YEAR_MANUF', 'LIGHT_CONDITION', 
                   'ROAD_GEOMETRY', 'SPEED_ZONE', 'SEX', 'AGE_GROUP', 'LICENCE_STATE']

X = df[feature_columns].copy()
y = df['severity']

# Handle missing values
imputer_mean = SimpleImputer(strategy='mean')
imputer_mode = SimpleImputer(strategy='most_frequent')

# Identify numerical and categorical columns
numerical_cols = ['VEHICLE_YEAR_MANUF', 'AGE_GROUP']
categorical_cols = ['VEHICLE_BODY_STYLE', 'LIGHT_CONDITION', 'ROAD_GEOMETRY', 
                    'SPEED_ZONE', 'SEX', 'LICENCE_STATE']

# Apply imputation
X[numerical_cols] = imputer_mean.fit_transform(X[numerical_cols])
X[categorical_cols] = imputer_mode.fit_transform(X[categorical_cols])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# Feature scaling for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class weights for imbalanced dataset
class_weights = class_weight.compute_class_weight('balanced', 
                                                  classes=np.unique(y_train), 
                                                  y=y_train)
class_weight_dict = dict(enumerate(class_weights))

print("Class distribution in training set:")
print(y_train.value_counts().sort_index())
print("\nClass weights:", class_weight_dict)

# Model 1: Decision Tree Classifier
print("\n=== Decision Tree Classifier ===")

# Simple hyperparameter tuning (reduced parameters for faster execution)
dt_params = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 10],
    'criterion': ['gini', 'entropy']
}

dt_classifier = DecisionTreeClassifier(random_state=42, class_weight=class_weight_dict)

# Grid search with n_jobs=1 to avoid parallel processing issues
dt_grid_search = GridSearchCV(dt_classifier, dt_params, 
                              cv=3, scoring='f1_weighted', n_jobs=1)  # Changed n_jobs to 1
dt_grid_search.fit(X_train, y_train)

# Best model
best_dt = dt_grid_search.best_estimator_
print(f"Best parameters: {dt_grid_search.best_params_}")

# Predictions
dt_pred = best_dt.predict(X_test)

# Evaluation
print("\nDecision Tree Performance:")
print(f"Accuracy: {accuracy_score(y_test, dt_pred):.4f}")
print(f"Weighted F1-score: {f1_score(y_test, dt_pred, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, dt_pred, 
                          target_names=['Low', 'Medium', 'High']))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm_dt = confusion_matrix(y_test, dt_pred)

# Create heatmap
im = plt.imshow(cm_dt, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Decision Tree Confusion Matrix')
plt.colorbar(im)

# Add text annotations
thresh = cm_dt.max() / 2.
for i in range(cm_dt.shape[0]):
    for j in range(cm_dt.shape[1]):
        plt.text(j, i, format(cm_dt[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm_dt[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(range(3), ['Low', 'Medium', 'High'])
plt.yticks(range(3), ['Low', 'Medium', 'High'])
plt.tight_layout()
plt.savefig('dt_confusion_matrix.png')
plt.close()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': best_dt.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Feature Importance')
plt.title('Decision Tree Feature Importance')
plt.tight_layout()
plt.savefig('dt_feature_importance.png')
plt.close()

print("\nFeature Importance:")
print(feature_importance)

# Model 2: K-Nearest Neighbors Classifier
print("\n=== K-Nearest Neighbors Classifier ===")

# Simple hyperparameter tuning
knn_params = {
    'n_neighbors': [5, 9, 15],
    'weights': ['uniform', 'distance']
}

knn_classifier = KNeighborsClassifier()

# Grid search with n_jobs=1
knn_grid_search = GridSearchCV(knn_classifier, knn_params, 
                               cv=3, scoring='f1_weighted', n_jobs=1)  # Changed n_jobs to 1
knn_grid_search.fit(X_train_scaled, y_train)

# Best model
best_knn = knn_grid_search.best_estimator_
print(f"Best parameters: {knn_grid_search.best_params_}")

# Predictions
knn_pred = best_knn.predict(X_test_scaled)

# Evaluation
print("\nKNN Performance:")
print(f"Accuracy: {accuracy_score(y_test, knn_pred):.4f}")
print(f"Weighted F1-score: {f1_score(y_test, knn_pred, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, knn_pred, 
                          target_names=['Low', 'Medium', 'High']))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm_knn = confusion_matrix(y_test, knn_pred)

# Create heatmap
im = plt.imshow(cm_knn, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('KNN Confusion Matrix')
plt.colorbar(im)

# Add text annotations
thresh = cm_knn.max() / 2.
for i in range(cm_knn.shape[0]):
    for j in range(cm_knn.shape[1]):
        plt.text(j, i, format(cm_knn[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm_knn[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(range(3), ['Low', 'Medium', 'High'])
plt.yticks(range(3), ['Low', 'Medium', 'High'])
plt.tight_layout()
plt.savefig('knn_confusion_matrix.png')
plt.close()

# Model Comparison
print("\n=== Model Comparison ===")
comparison_df = pd.DataFrame({
    'Model': ['Decision Tree', 'KNN'],
    'Accuracy': [accuracy_score(y_test, dt_pred), accuracy_score(y_test, knn_pred)],
    'Weighted F1': [f1_score(y_test, dt_pred, average='weighted'), 
                    f1_score(y_test, knn_pred, average='weighted')]
})

print(comparison_df)

# Plot comparison
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy comparison
ax[0].bar(comparison_df['Model'], comparison_df['Accuracy'])
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Model Accuracy Comparison')
ax[0].set_ylim(0, 1)

# F1-score comparison
ax[1].bar(comparison_df['Model'], comparison_df['Weighted F1'])
ax[1].set_ylabel('Weighted F1-score')
ax[1].set_title('Model F1-score Comparison')
ax[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Save the trained models
import pickle

with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(best_dt, f)

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(best_knn, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModels saved successfully!")
print("\nAnalysis Complete! Check the generated plots and saved models.")
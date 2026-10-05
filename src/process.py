import pandas as pd
import numpy as np

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import mlflow

df = pd.read_csv("../data/data.csv", sep=';')

df = df.drop(columns=['id'], errors='ignore')

X = df.drop(columns=['Target'])
y = df['Target']

le = LabelEncoder()
y = le.fit_transform(y)

X = pd.get_dummies(X, columns=['Course'], drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
sm = SMOTE()
X_train, y_train = sm.fit_resample(X_train, y_train)

# Initial Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Model trained successfully....", model.get_params())

# 1. Simple Text/Array format mein print karne ke liye
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# 2. Color-coded Heatmap (Graph)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Dropout', 'Enrolled', 'Graduate'], 
            yticklabels=['Dropout', 'Enrolled', 'Graduate'])
plt.title('Confusion Matrix: Student Risk Prediction')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
# plt.show() 

# MLflow Tracking Setup
mlflow.set_experiment("Student_Risk_Prediction")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

print("Running GridSearchCV, please wait...")
grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid,
                    cv=3,
                    scoring='accuracy',
                    n_jobs=-1)

grid.fit(X_train, y_train)

# MLflow Logging
with mlflow.start_run():
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("best_accuracy", grid.best_score_)
    

    mlflow.sklearn.log_model(
        grid.best_estimator_, 
        "model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\n--- Best Model Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(best_model, "model.pkl")
print("Model successfully saved as model.pkl")
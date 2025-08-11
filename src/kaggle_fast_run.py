from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

# load the preprocessed dataset 

loan_applications_df_encoded=pd.read_csv("data/processed/kaggle_clean.csv")
# Split features and target
X = loan_applications_df_encoded.drop(columns=['isDefault'])
y = loan_applications_df_encoded['isDefault']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train_scaled, y_train)

# Evaluate
y_pred = logreg.predict(X_test_scaled)
y_probs = logreg.predict_proba(X_test_scaled)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", round(roc_auc_score(y_test, y_probs), 3))

# ROC Curve
RocCurveDisplay.from_estimator(logreg, X_test_scaled, y_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import pandas as pd

# Assuming X and y are already defined
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train the Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))

# Plot ROC Curve
RocCurveDisplay.from_estimator(rf_model, X_test, y_test)
plt.show()



from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np

# Define hyperparameter grid for Random Forest
rf_params = {
    'n_estimators': [100, 150],         # Number of trees in the forest
    'max_depth': [None, 10, 20],        # Maximum depth of the tree
    'min_samples_split': [2, 4]         # Minimum samples required to split a node
}

# Perform Grid Search with 3-fold cross-validation
rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_params,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train, y_train)

# Retrieve the best model
best_rf_model = rf_grid.best_estimator_
print("RF Parameters:", rf_grid.best_params_)

# Evaluate tuned Random Forest
y_pred_rf = best_rf_model.predict(X_test)
y_prob_rf = best_rf_model.predict_proba(X_test)[:, 1]

print("\n📈 Tuned RF Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Tuned RF ROC AUC Score:", round(roc_auc_score(y_test, y_prob_rf), 3))

# Plot ROC Curve
RocCurveDisplay.from_estimator(best_rf_model, X_test, y_test)
plt.title("Tuned Random Forest - ROC Curve")
plt.show()



from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize & Train
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Evaluation
print("Confusion Matrix:")
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print(cm_xgb)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

auc_xgb = roc_auc_score(y_test, y_prob_xgb)
print(f"ROC AUC Score: {auc_xgb:.4f}")

# ROC Curve
RocCurveDisplay.from_estimator(xgb_model, X_test, y_test)
plt.title("XGBoost - ROC Curve")
plt.show()



from xgboost import XGBClassifier

# Define hyperparameter grid for XGBoost
xgb_params = {
    'n_estimators': [100, 150],         # Number of boosting rounds
    'max_depth': [3, 5],                # Maximum tree depth
    'learning_rate': [0.1, 0.05]        # Step size shrinkage
}

# Perform Grid Search with 3-fold cross-validation
xgb_grid = GridSearchCV(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_grid=xgb_params,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=1
)
xgb_grid.fit(X_train, y_train)

# Retrieve the best model
best_xgb_model = xgb_grid.best_estimator_
print("Best XGB Parameters:", xgb_grid.best_params_)

# Evaluate tuned XGBoost model
y_pred_xgb = best_xgb_model.predict(X_test)
y_prob_xgb = best_xgb_model.predict_proba(X_test)[:, 1]

print("\n Tuned XGB Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print(" Tuned XGB ROC AUC Score:", round(roc_auc_score(y_test, y_prob_xgb), 3))

# Plot ROC Curve
RocCurveDisplay.from_estimator(best_xgb_model, X_test, y_test)
plt.title("Tuned XGBoost - ROC Curve")
plt.show()

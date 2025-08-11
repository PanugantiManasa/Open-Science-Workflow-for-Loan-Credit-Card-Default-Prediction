#load the preprocessed dataset
credit_card_df_encoded = pd.read_csv("data/processed/uci_clean.csv")

# Separate features and target
X = credit_card_df_encoded.drop(columns=['default payment next month'])
y = credit_card_df_encoded['default payment next month']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#  Train baseline logistic regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

#  Predict and evaluate
y_pred = logreg.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Retrain logistic regression
logreg_scaled = LogisticRegression(max_iter=2000)
logreg_scaled.fit(X_train_scaled, y_train)

# Predict class labels and probabilities
y_pred_scaled = logreg_scaled.predict(X_test_scaled)
y_probs = logreg_scaled.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_scaled))
print("\nClassification Report:\n", classification_report(y_test, y_pred_scaled))
print("ROC AUC Score:", round(roc_auc_score(y_test, y_probs), 3))

# Plot ROC Curve
RocCurveDisplay.from_estimator(logreg_scaled, X_test_scaled, y_test)
plt.title("Logistic Regression (Scaled) - ROC Curve")
plt.show()

from sklearn.ensemble import RandomForestClassifier

# Train Random Forest with balanced class weights
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf_model.predict(X_test)
y_probs_rf = rf_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

#  ROC AUC
auc_rf = roc_auc_score(y_test, y_probs_rf)
print("Random Forest ROC AUC Score:", round(auc_rf, 3))

# Plot ROC
RocCurveDisplay.from_estimator(rf_model, X_test, y_test)

import matplotlib.pyplot as plt
import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np

# Define hyperparameter grid for Random Forest
rf_params = {
    'n_estimators': [100, 150],         # Number of trees
    'max_depth': [None, 10, 20],        # Maximum depth
    'min_samples_split': [2, 4]         # Minimum number of samples to split
}

# Grid search with cross-validation
rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=rf_params,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train, y_train)

# Extract best model
best_rf_model = rf_grid.best_estimator_
print("RF Parameters:", rf_grid.best_params_)

# Predict and evaluate
y_pred_rf = best_rf_model.predict(X_test)
y_prob_rf = best_rf_model.predict_proba(X_test)[:, 1]

print("\n Tuned RF Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Tuned RF ROC AUC Score:", round(roc_auc_score(y_test, y_prob_rf), 3))

# ROC Curve
RocCurveDisplay.from_estimator(best_rf_model, X_test, y_test)
plt.title("Tuned Random Forest - ROC Curve")
plt.show()



import xgboost as xgb
from xgboost import XGBClassifier

# Compute scale_pos_weight to address imbalance
scale_ratio = (y_train == 0).sum() / (y_train == 1).sum()

# Train XGBoost model
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_ratio,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_xgb = xgb_model.predict(X_test)
y_probs_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))
print("XGBoost ROC AUC Score:", round(roc_auc_score(y_test, y_probs_xgb), 3))

#  Plot ROC
RocCurveDisplay.from_estimator(xgb_model, X_test, y_test)


from xgboost import XGBClassifier

# Define hyperparameter grid for XGBoost
xgb_params = {
    'n_estimators': [100, 150],         # Number of boosting rounds
    'max_depth': [3, 5],                # Maximum depth
    'learning_rate': [0.1, 0.05]        # Learning rate (shrinkage)
}

# Compute imbalance ratio
scale_ratio = (y_train == 0).sum() / (y_train == 1).sum()

# Grid search with cross-validation
xgb_grid = GridSearchCV(
    estimator=XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_ratio,
        random_state=42
    ),
    param_grid=xgb_params,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=1
)
xgb_grid.fit(X_train, y_train)

# Extract best model
best_xgb_model = xgb_grid.best_estimator_
print("XGB Parameters:", xgb_grid.best_params_)

# Predict and evaluate
y_pred_xgb = best_xgb_model.predict(X_test)
y_prob_xgb = best_xgb_model.predict_proba(X_test)[:, 1]

print("\n Tuned XGB Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print(" Tuned XGB ROC AUC Score:", round(roc_auc_score(y_test, y_prob_xgb), 3))

# ROC Curve
RocCurveDisplay.from_estimator(best_xgb_model, X_test, y_test)
plt.title("Tuned XGBoost - ROC Curve")
plt.show()
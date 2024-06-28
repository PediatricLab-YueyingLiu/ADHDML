import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import shap

# Set working directory and load libraries
import os
os.chdir("your path here")

# Read the data
data = pd.read_csv("your own csv")

# Ensure 'Group' is the target column
target_column = 'your Group column here'

# Set features and labels
X = data.drop(columns=target_column)
y = data[target_column]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=121, stratify=y)

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), X.columns)
    ]
)

# Define the models
models = {
    'Lasso Regression': LogisticRegression(max_iter=10000, solver='saga', penalty='l1'),
    'Random Forest': RandomForestClassifier(random_state=121),
    'AdaBoost': AdaBoostClassifier(random_state=121),
    'GradientBoosting': GradientBoostingClassifier(random_state=121),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', seed=121)
}

# Train and evaluate the models
results = {}
plt.figure(figsize=(20, 10))
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict_proba(X_train)[:, 1]
    y_test_pred = pipeline.predict_proba(X_test)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    test_acc = accuracy_score(y_test, pipeline.predict(X_test))
    
    results[name] = {
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_acc': train_acc,
        'test_acc': test_acc
    }
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_pred)
    roc_auc = auc(fpr, tpr)
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    # Plot PRC curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_pred)
    prc_auc = auc(recall, precision)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'{name} (AUC = {prc_auc:.2f})')

# Set ROC curve settings
plt.subplot(1, 2, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

# Set PRC curve settings
plt.subplot(1, 2, 2)
plt.plot([0, 1], [1, 0], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')

plt.show()

# List of feature counts
feature_counts = [3, 8, 13, 18, 23, 28, 33]

# Train and evaluate models with different feature counts
results = {model: [] for model in models}

for k in feature_counts:
    selector = SelectKBest(f_classif, k=k)
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('selector', selector),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        y_test_pred = pipeline.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_pred)
        results[name].append(test_auc)

# Plot AUC vs number of features
plt.figure(figsize=(10, 6))
for model, auc_values in results.items():
    plt.plot(feature_counts, auc_values, marker='o', label=model)

plt.axvline(x=8, color='k', linestyle='--', label='Selected number of features')
plt.xlabel('Numbers of features')
plt.ylabel('Area under the ROC')
plt.title('Feature Reduction')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Use XGBoost model and SHAP values for model interpretation
# Select top 8 features
selector = SelectKBest(f_classif, k=8)
X_train_selected = selector.fit_transform(preprocessor.fit_transform(X_train), y_train)
X_test_selected = selector.transform(preprocessor.transform(X_test))

# Retrain XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', seed=121)
model.fit(X_train_selected, y_train)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train_selected)

# Get selected feature names
selected_features = X.columns[selector.get_support()]

# Plot SHAP summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train_selected, feature_names=selected_features)
plt.show()

# Plot SHAP bar plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train_selected, feature_names=selected_features, plot_type="bar")
plt.show()

# Plot SHAP dependence plot
for feature in selected_features:
    plt.figure()
    shap.dependence_plot(feature, shap_values, X_train_selected, feature_names=selected_features, interaction_index=None, show=False)
    plt.title(f'SHAP Dependence Plot for {feature}')
    plt.show()
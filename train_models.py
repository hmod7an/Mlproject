"""
Train ML Models for Tailoring Dashboard
Uses the enhanced dataset with improved satisfaction logic
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load enhanced dataset
print("Loading enhanced dataset...")
df = pd.read_csv('/home/ubuntu/tailoring_dashboard/enhanced_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"Satisfaction distribution:\n{df['Satisfaction'].value_counts()}")

# ============================================
# FEATURE ENGINEERING
# ============================================
print("\nPreparing features...")

# Select features for modeling
feature_cols = [
    'order_Type', 'Tailoring_Style', 'size', 'order_Quantity',
    'length_cm', 'width_cm', 'sleeve_cm', 'fabric_meters',
    'Price_Per_Unit', 'Fabric_Price_Per_Meter', 'order_Tax',
    'order_Discount', 'Expected_Delivery_Days', 'Days_Difference'
]

# Create a copy for modeling
model_df = df[feature_cols + ['Total_Amount', 'Satisfaction']].copy()

# Handle missing values
model_df = model_df.dropna()

# Encode categorical variables
label_encoders = {}
categorical_cols = ['order_Type', 'Tailoring_Style', 'size']

for col in categorical_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))
    label_encoders[col] = le

# Save label encoders
joblib.dump(label_encoders, '/home/ubuntu/tailoring_dashboard/label_encoders.pkl')

# Prepare features
X = model_df.drop(['Total_Amount', 'Satisfaction'], axis=1)
y_satisfaction = model_df['Satisfaction']
y_amount = model_df['Total_Amount']
y_length = model_df['length_cm']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, '/home/ubuntu/tailoring_dashboard/scaler.pkl')

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, '/home/ubuntu/tailoring_dashboard/feature_names.pkl')

# ============================================
# TRAIN SATISFACTION MODEL
# ============================================
print("\n" + "="*50)
print("Training Satisfaction Model (Classification)")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_satisfaction, test_size=0.2, random_state=42, stratify=y_satisfaction
)

# Random Forest Classifier
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'  # Handle imbalance
)
rf_clf.fit(X_train, y_train)

# Evaluate
y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nSatisfaction Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Unsatisfied', 'Satisfied']))

# Cross-validation
cv_scores = cross_val_score(rf_clf, X_scaled, y_satisfaction, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")

# Save model
joblib.dump(rf_clf, '/home/ubuntu/tailoring_dashboard/satisfaction_model.pkl')

# ============================================
# TRAIN PRICE MODEL
# ============================================
print("\n" + "="*50)
print("Training Price Model (Regression)")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_amount, test_size=0.2, random_state=42
)

rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42
)
rf_reg.fit(X_train, y_train)

y_pred = rf_reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nPrice Model R²: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f} SAR")

joblib.dump(rf_reg, '/home/ubuntu/tailoring_dashboard/price_model.pkl')

# ============================================
# TRAIN LENGTH MODEL
# ============================================
print("\n" + "="*50)
print("Training Length Model (Regression)")
print("="*50)

# Use only relevant features for length
length_features = ['size', 'width_cm', 'sleeve_cm', 'fabric_meters']
X_length = model_df[length_features]
X_length_scaled = StandardScaler().fit_transform(X_length)

X_train, X_test, y_train, y_test = train_test_split(
    X_length_scaled, y_length, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nLength Model R²: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f} cm")

joblib.dump(lr, '/home/ubuntu/tailoring_dashboard/length_model.pkl')

# ============================================
# SHAP ANALYSIS
# ============================================
print("\n" + "="*50)
print("Generating SHAP Analysis")
print("="*50)

# SHAP for Satisfaction Model
print("Creating SHAP plots for Satisfaction model...")
explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(X_scaled[:100])

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values[1], X_scaled[:100], feature_names=feature_names, show=False)
plt.title('SHAP Summary - Customer Satisfaction', fontsize=14)
plt.tight_layout()
plt.savefig('/home/ubuntu/tailoring_dashboard/shap_satisfaction.png', dpi=150, bbox_inches='tight')
plt.close()

# SHAP for Price Model
print("Creating SHAP plots for Price model...")
explainer_price = shap.TreeExplainer(rf_reg)
shap_values_price = explainer_price.shap_values(X_scaled[:100])

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_price, X_scaled[:100], feature_names=feature_names, show=False)
plt.title('SHAP Summary - Price Prediction', fontsize=14)
plt.tight_layout()
plt.savefig('/home/ubuntu/tailoring_dashboard/shap_price.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature Importance Bar Plot
plt.figure(figsize=(10, 6))
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=True)

plt.barh(importance['feature'], importance['importance'], color='#1A237E')
plt.xlabel('Feature Importance')
plt.title('Feature Importance - Satisfaction Model')
plt.tight_layout()
plt.savefig('/home/ubuntu/tailoring_dashboard/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ All models trained and saved!")
print("\nModel files saved:")
print("- satisfaction_model.pkl")
print("- price_model.pkl")
print("- length_model.pkl")
print("- label_encoders.pkl")
print("- scaler.pkl")
print("- feature_names.pkl")
print("\nSHAP plots saved:")
print("- shap_satisfaction.png")
print("- shap_price.png")
print("- feature_importance.png")

import numpy as np
import random
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.metrics import ConfusionMatrixDisplay

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Load dataset
df = pd.read_csv('/content/drive/MyDrive/dementia_dataset.csv')

# Handle missing data using mean imputation
imputer = SimpleImputer(strategy='mean')
df[['SES', 'MMSE']] = imputer.fit_transform(df[['SES', 'MMSE']])

# Label encoding for categorical variables
encoder = LabelEncoder()
df['Encoded_Group'] = encoder.fit_transform(df['Group'])

# Drop unnecessary columns
columns_to_drop = ['Subject ID', 'MRI ID', 'Visit', 'MR Delay']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Drop additional columns not needed for the model
df.drop(columns=['M/F', 'Hand', 'SES'], inplace=True)

# Feature and target separation
X = df.drop(columns=['Group', 'Encoded_Group'])  # Features
y = df['Encoded_Group']  # Target

# Feature scaling using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection using RFE
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=6)
X_rfe = rfe_selector.fit_transform(X_scaled, y)

# PCA for dimensionality reduction
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_scaled)

# Use SMOTE to handle class imbalance
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize KFold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize and fit the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_train_pred = rf_classifier.predict(X_train)
y_test_pred = rf_classifier.predict(X_test)

print(f"Random Forest Test Accuracy: {accuracy_score(y_test, y_test_pred) * 100:.2f}%")

print(f"\nClassification Report for Random Forest:")
print(classification_report(y_test, y_test_pred))

print(f"\nConfusion Matrix for Random Forest:")
print(confusion_matrix(y_test, y_test_pred))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['Demented', 'Nondemented', 'Converted'])
disp.plot(cmap=plt.cm.Blues)
plt.show()






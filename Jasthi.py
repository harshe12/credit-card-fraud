#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


# In[2]:


# Load the dataset
data = pd.read_csv('CreditCardData.csv')


# In[3]:


data


# In[4]:


# Display basic information about the dataset
print(data.info())
print(data.describe())


# In[5]:


# Drop any non-numeric columns or columns that are not relevant for modeling
data = data.drop(['Transaction ID', 'Date', 'Shipping Address'], axis=1)


# In[6]:


# Handle missing values
data['Amount'] = data['Amount'].str.replace('£', '').astype(float)
data = data.dropna()  # Drop rows with missing values


# In[8]:


# Display the DataFrame after handling duplicates
print("DataFrame after removing duplicate values:")
data


# In[27]:


# Encode categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['Day of Week', 'Time', 'Age', 'Entry Mode', 'Fraud', 'Type of Card', 'Type of Transaction', 'Merchant Group', 'Country of Transaction', 'Country of Residence', 'Gender', 'Bank']
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])


# In[12]:


# Visualize correlation matrix for numeric columns only
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 7))
sns.heatmap(numeric_data.corr(), annot=True, vmin=-1, vmax=1, cmap='magma')
plt.show()


# In[11]:


# Visualize boxplot
plt.figure(figsize=(15, 10))
sns.boxplot(data=data, orient='h')
plt.show()


# In[13]:


# Visualize pairplot
plt.figure(figsize=(13, 17))
sns.pairplot(data=data)
plt.show()


# In[65]:


print(data.columns)


# In[66]:


plt.figure(figsize=(12, 8))
sns.violinplot(x='Day of Week', y='Age', data=data)
plt.title('Violin Plot of Age by Day of Week')
plt.show()


# In[72]:


plt.figure(figsize=(8, 5))
sns.countplot(x='Amount', data=data)
plt.title('Count Plot of Amount')
plt.show()


# In[73]:


sns.jointplot(x='Time', y='Amount', data=data, kind='scatter')
plt.show()


# In[28]:


# Define numerical and categorical features
numeric_features = ['Time', 'Entry Mode', 'Fraud', 'Age']
categorical_features = ['Day of Week', 'Type of Card', 'Type of Transaction', 'Merchant Group', 'Country of Transaction', 'Country of Residence', 'Gender', 'Bank']


# In[29]:


# Create transformers for numerical and categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# In[30]:


# Create a pipeline with preprocessing and logistic regression
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', LogisticRegression())])


# In[31]:


# Segmenting the data into features (X) and target variable (y)
features = data.drop('Fraud', axis=1)
target = data['Fraud']


# In[32]:


# Dividing the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# In[45]:


# Constructing a Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)

print(f'Logistic Regression Accuracy: {logistic_accuracy:.4f}')


# In[46]:


# Building a Decision Tree Classifier model
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_predictions)

print(f'Decision Tree Classifier Accuracy: {tree_accuracy:.4f}')


# In[47]:


# Generating the Confusion Matrix for Decision Tree Classifier
tree_confusion_matrix = confusion_matrix(y_test, tree_predictions)
print('Confusion Matrix for Decision Tree Classifier:')
print(tree_confusion_matrix)


# In[48]:


# Generating the Confusion Matrix for Logistic Regression
logistic_confusion_matrix = confusion_matrix(y_test, logistic_predictions)
print('Confusion Matrix for Logistic Regression:')
print(logistic_confusion_matrix)


# In[49]:


# Producing the Classification Report for Decision Tree Classifier
report_tree = classification_report(y_test, tree_predictions)
print('Classification Report for Decision Tree Classifier:')
print(report_tree)


# In[50]:


# Producing the Classification Report for Logistic Regression
report_logistic = classification_report(y_test, logistic_predictions)
print('Classification Report for Logistic Regression:')
print(report_logistic)


# In[51]:


# Visualizing the Confusion Matrix for Decision Tree Classifier
plt.figure(figsize=(8, 6))
sns.heatmap(tree_confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Decision Tree Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[52]:


# Visualizing the Confusion Matrix for Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(logistic_confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[59]:


# ROC Curves
plt.figure(figsize=(8, 6))
# Logistic Regression ROC Curve
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, logistic_model.predict_proba(X_test)[:, 1])
roc_auc_logistic = auc(fpr_logistic, tpr_logistic)

plt.plot(fpr_logistic, tpr_logistic, label=f'Logistic Regression (AUC = {roc_auc_logistic:.2f})')
# Decision Tree Classifier ROC Curve
fpr_tree, tpr_tree, _ = roc_curve(y_test, tree_model.predict_proba(X_test)[:, 1])
roc_auc_tree = auc(fpr_tree, tpr_tree)

plt.plot(fpr_tree, tpr_tree, label=f'Decision Tree Classifier (AUC = {roc_auc_tree:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

plt.show()


# In[60]:


from sklearn.metrics import precision_recall_curve

# Assuming 'logistic_model' is your Logistic Regression model
precision, recall, thresholds = precision_recall_curve(y_test, logistic_model.predict_proba(X_test)[:, 1])

plt.plot(recall, precision, label='Logistic Regression')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Logistic Regression')
plt.legend()
plt.show()


# In[ ]:





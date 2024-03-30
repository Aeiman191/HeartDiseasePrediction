import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text,  plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle


df = pd.read_csv('heart_v2.csv')
data_test = df['heart disease']
data_train = df.drop('heart disease', axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data_train, data_test, test_size=0.2, random_state=42)


# Train the decision tree classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
# Predict the labels of the test set
y_pred_dt = dt.predict(X_test)
# Calculate the accuracy of the classifier
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = metrics.precision_score(y_test, y_pred_dt)
recall_dt = metrics.recall_score(y_test, y_pred_dt)
f1_dt = metrics.f1_score(y_test, y_pred_dt)
print("Decision Tree Classifier" + "\n")
print("Accuracy:", accuracy_dt)
print("Precision:", precision_dt)
print("Recall:", recall_dt)
print("F1:", f1_dt)

filename = 'DecisionTreeClassifier_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(dt, file)
    
# Apply SVM classifier

svc = SVC()
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svc)
precision_svm = metrics.precision_score(y_test, y_pred_svc)
recall_svm = metrics.recall_score(y_test, y_pred_svc)
f1_svm = metrics.f1_score(y_test, y_pred_svc)
print("\nSupport Vector Machine Classifier" + "\n")
print("Accuracy: ", accuracy_svm)
print("Precision: ", precision_svm)
print("Recall: ", recall_svm)
print("F1: ", f1_svm)



filename = 'SVC_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(svc, file)


# Apply SVM classifier

svm_f = SVC(kernel='linear', C=1.0,gamma='auto' ,random_state=42)
svm_f.fit(X_train, y_train)
y_pred_svm_f = svm_f.predict(X_test)
accuracy_svm_f = accuracy_score(y_test, y_pred_svm_f)
precision_svm_f = metrics.precision_score(y_test, y_pred_svm_f)
recall_svm_f = metrics.recall_score(y_test, y_pred_svm_f)
f1_svm_f = metrics.f1_score(y_test, y_pred_svm_f)
print("\n\nSupport Vector Machine with linear kernel" + "\n")
print("Accuracy: ", accuracy_svm_f)
print("Precision: ", precision_svm_f)
print("Recall: ", recall_svm_f)
print("F1: ", f1_svm_f)



filename = 'SVC_model_2.pkl'
with open(filename, 'wb') as file:
    pickle.dump(svm_f, file)



# Create an instance of the logistic regression model
lr = LogisticRegression()
# Fit the model to the training data
lr.fit(X_train, y_train)
# Perform prediction using logistic regression
y_pred_lr = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print("Logistic Regression" + "\n")
print("Accuracy:", accuracy_lr)
print("Precision:", precision_lr)
print("Recall:", recall_lr)
print("F1:", f1_lr)

filename = 'logistic_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(lr, file)
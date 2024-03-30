import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle

# Load the dataset
df = pd.read_csv('heart_v2.csv')
data_test = df['heart disease']
data_train = df.drop('heart disease', axis=1)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data_train, data_test, test_size=0.2, random_state=42)

# Train the decision tree classifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
print("Decision Tree Classifier\n")
print(f"Accuracy: {accuracy_dt}")
print(f"Precision: {precision_dt}")
print(f"Recall: {recall_dt}")
print(f"F1: {f1_dt}")

FILENAME_DT = 'DecisionTreeClassifier_model.pkl'
with open(FILENAME_DT, 'wb') as file:
    pickle.dump(dt, file)

# Apply SVM classifier
svc = SVC()
svc.fit(x_train, y_train)
y_pred_svc = svc.predict(x_test)
accuracy_svm = accuracy_score(y_test, y_pred_svc)
precision_svm = precision_score(y_test, y_pred_svc)
recall_svm = recall_score(y_test, y_pred_svc)
f1_svm = f1_score(y_test, y_pred_svc)
print("\nSupport Vector Machine Classifier\n")
print(f"Accuracy: {accuracy_svm}")
print(f"Precision: {precision_svm}")
print(f"Recall: {recall_svm}")
print(f"F1: {f1_svm}")

FILENAME_SVC = 'SVC_model.pkl'
with open(FILENAME_SVC, 'wb') as file:
    pickle.dump(svc, file)

# Apply SVM classifier with linear kernel
svm_f = SVC(kernel='linear', C=1.0, gamma='auto', random_state=42)
svm_f.fit(x_train, y_train)
y_pred_svm_f = svm_f.predict(x_test)
accuracy_svm_f = accuracy_score(y_test, y_pred_svm_f)
precision_svm_f = precision_score(y_test, y_pred_svm_f)
recall_svm_f = recall_score(y_test, y_pred_svm_f)
f1_svm_f = f1_score(y_test, y_pred_svm_f)
print("\nSupport Vector Machine with linear kernel\n")
print(f"Accuracy: {accuracy_svm_f}")
print(f"Precision: {precision_svm_f}")
print(f"Recall: {recall_svm_f}")
print(f"F1: {f1_svm_f}")

FILENAME_SVC_LINEAR = 'SVC_model_2.pkl'
with open(FILENAME_SVC_LINEAR, 'wb') as file:
    pickle.dump(svm_f, file)

lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print("Logistic Regression" + "\n")
print("Accuracy:", accuracy_lr)
print("Precision:", precision_lr)
print("Recall:", recall_lr)
print("F1:", f1_lr)

FILENAME_LR = 'logistic_regression_model.pkl'
with open(FILENAME_LR, 'wb') as file:
    pickle.dump(lr, file)

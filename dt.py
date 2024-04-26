import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("pima_indians_diabetes.csv")

# Split data into input and target variable(s)
X = data.drop("class", axis=1)
y = data["class"]

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, test_size=0.10, random_state=42
)

# Create and train the classifier
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Prediction on both train and test sets
y_train_pred = classifier.predict(X_train)
y_test_pred = classifier.predict(X_test)

# Model Evaluation
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Compute F1-score for training set
train_f1 = f1_score(y_train, y_train_pred)

# Compute F1-score for testing set
test_f1 = f1_score(y_test, y_test_pred)

print("F1 Score for Training Data:", train_f1)
print("F1 Score for Testing Data:", test_f1)

print("Classification Report for Training Data:")
print(classification_report(y_train, y_train_pred))
print("Confusion Matrix for Training Data:")
print(confusion_matrix(y_train, y_train_pred))

print("Classification Report for Testing Data:")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix for Testing Data:")
print(confusion_matrix(y_test, y_test_pred))

# Precision-Recall curve for test set
precision, recall, _ = precision_recall_curve(y_test, classifier.predict_proba(X_test)[:, 1])
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()

# ROC curve for test set
fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

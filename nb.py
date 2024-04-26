# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
from pydataset import data

# Load Iris dataset
df = data('iris')

# Split dataset into features and target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split dataset into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predict the target variable on the test set
y_pred = nb_classifier.predict(X_test)

# Calculate metrics
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(np.unique(y))):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred, pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure()
for i in range(len(np.unique(y))):
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Compute Precision-Recall curve and area for each class
precision = dict()
recall = dict()
pr_auc = dict()
for i in range(len(np.unique(y))):
    precision[i], recall[i], _ = precision_recall_curve(y_test, y_pred, pos_label=i)
    pr_auc[i] = auc(recall[i], precision[i])

# Plot Precision-Recall curve
plt.figure()
for i in range(len(np.unique(y))):
    plt.plot(recall[i], precision[i], label='PR curve (area = %0.2f)' % pr_auc[i])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()

# Print results
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy Score:", accuracy)
print("F1 Score:", f1)

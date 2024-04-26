import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, f1_score, precision_recall_curve

# Loading Dataset
data = pd.read_csv("Churn_Modelling.csv")

# Generating Independent Variable Matrix and Dependent Variable Vector
X = data.iloc[:, 3:-1].values
y = data.iloc[:, -1].values

# Encoding Categorical Variable Gender
label_encoder = LabelEncoder()
X[:, 2] = label_encoder.fit_transform(X[:, 2])

# Encoding Categorical Variable Geography
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder="passthrough")
X = ct.fit_transform(X)

# Splitting dataset into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Performing Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initializing ANN
ann = Sequential()

# Adding Input Layer and First Hidden Layer
ann.add(Dense(units=8, activation="relu"))

# Adding Second Hidden Layer
ann.add(Dense(units=6, activation="relu"))

# Adding Output Layer
ann.add(Dense(units=1, activation="sigmoid"))

# Compiling ANN
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

# Fitting ANN and tracking training history
history = ann.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

# Predicting results for the testing data
y_pred = ann.predict(X_test)
y_pred_binary = (y_pred > 0.5)

# Performance Metrics
print("Classification Report:")
print(classification_report(y_test, y_pred_binary))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_binary))

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy Score:", accuracy)

# F1 Score
f1 = f1_score(y_test, y_pred_binary)
print("F1 Score:", f1)

# Precision-Recall curve for test set
precision, recall, _ = precision_recall_curve(y_test, y_pred)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plotting ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Visualizing Training History
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

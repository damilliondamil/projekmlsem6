from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load data hasil ekstraksi fitur fft
x = pd.read_csv("data/feature_VBL-VA001.csv", header=None)

# Load label
y = pd.read_csv("data/label_VBL-VA001.csv", header=None)

# Make 1D array to avoid warning
y = pd.Series.ravel(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True
)

print("Shape of Train Data: {}".format(X_train.shape))
print("Shape of Test Data: {}".format(X_test.shape))

# Setup arrays to store training and test accuracies
n_estimators = np.arange(1, 100)
train_accuracy = np.empty(len(n_estimators))
test_accuracy = np.empty(len(n_estimators))

for i, n in enumerate(n_estimators):
    # Setup a random forest classifier with n_estimators
    rf_model = RandomForestClassifier(n_estimators=10)
    # Fit the model
    y_pred = rf_model.fit(X_train, y_train)
    # Compute accuracy on the training set
    train_accuracy[i] = rf_model.score(X_train, y_train)
    # Compute accuracy on the test set
    test_accuracy[i] = rf_model.score(X_test, y_test)

# Generate plot
plt.plot(n_estimators, test_accuracy, label='Testing Accuracy')
plt.plot(n_estimators, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.show()

# Print optimal n_estimators and max test accuracy
print(f"Optimal n_estimators: {np.argmax(test_accuracy)}")
print(f"Max test accuracy: {np.max(test_accuracy)}")

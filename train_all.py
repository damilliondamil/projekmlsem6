from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data hasil ekstraksi fitur fft
x = pd.read_csv("data/feature_VBL-VA001.csv", header=None)

# Load label
y = pd.read_csv("data/label_VBL-VA001.csv", header=None)

# Make 1D array to avoid warning
y = pd.Series.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True
)

print("Shape of Train Data: {}".format(X_train.shape))
print("Shape of Test Data: {}".format(X_test.shape))

# Support Vector Machine
c_svm = np.arange(1, 100)
train_accuracy_svm = np.empty(len(c_svm))
test_accuracy_svm = np.empty(len(c_svm))

for i, k in enumerate(c_svm):
    svm = SVC(C=k)
    svm.fit(X_train, y_train)
    train_accuracy_svm[i] = svm.score(X_train, y_train)
    test_accuracy_svm[i] = svm.score(X_test, y_test)

# K-Nearest Neighbors
neighbors = np.arange(1, 100)
train_accuracy_knn = np.empty(len(neighbors))
test_accuracy_knn = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracy_knn[i] = knn.score(X_train, y_train)
    test_accuracy_knn[i] = knn.score(X_test, y_test)

# Gaussian Naive Bayes
var_gnb = [10.0 ** i for i in np.arange(-1, -100, -1)]
train_accuracy_gnb = np.empty(len(var_gnb))
test_accuracy_gnb = np.empty(len(var_gnb))

for i, k in enumerate(var_gnb):
    model = GaussianNB(var_smoothing=k)
    gnb = model.fit(X_train, y_train)
    train_accuracy_gnb[i] = gnb.score(X_train, y_train)
    test_accuracy_gnb[i] = gnb.score(X_test, y_test)

# Random Forest
n_estimators = np.arange(1, 100)
train_accuracy_rf = np.empty(len(n_estimators))
test_accuracy_rf = np.empty(len(n_estimators))

for i, n in enumerate(n_estimators):
    rf = RandomForestClassifier(n_estimators=n)
    rf.fit(X_train, y_train)
    train_accuracy_rf[i] = rf.score(X_train, y_train)
    test_accuracy_rf[i] = rf.score(X_test, y_test)

# Plot results
plt.subplot(1, 4, 1)
plt.plot(c_svm, test_accuracy_svm, label='Testing Accuracy')
plt.plot(c_svm, train_accuracy_svm, label='Training Accuracy')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Support Vector Machine')

plt.subplot(1, 4, 2)
plt.plot(neighbors, test_accuracy_knn, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy_knn, label='Training Accuracy')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.title('K-Nearest Neighbors')

plt.subplot(1, 4, 3)
plt.plot(var_gnb, test_accuracy_gnb, label='Testing Accuracy')
plt.plot(var_gnb, train_accuracy_gnb, label='Training Accuracy')
plt.xlabel('var_smoothing')
plt.ylabel('Accuracy')
plt.title('Gaussian Naive Bayes')

plt.subplot(1, 4, 4)
plt.plot(n_estimators, test_accuracy_rf, label='Testing Accuracy')
plt.plot(n_estimators, train_accuracy_rf, label='Training Accuracy')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('Random Forest')

plt.legend()
plt.tight_layout()
plt.show()

# Print optimal parameters and maximum test accuracy for each model
print(f"Support Vector Machine: Optimal C: {c_svm[np.argmax(test_accuracy_svm)]}, Max test accuracy: {max(test_accuracy_svm)}")
print(f"K-Nearest Neighbors: Optimal n_neighbors: {neighbors[np.argmax(test_accuracy_knn)]}, Max test accuracy: {max(test_accuracy_knn)}")
print(f"Gaussian Naive Bayes: Optimal var_smoothing: {var_gnb[np.argmax(test_accuracy_gnb)]}, Max test accuracy: {max(test_accuracy_gnb)}")
print(f"Random Forest: Optimal n_estimators: {n_estimators[np.argmax(test_accuracy_rf)]}, Max test accuracy: {max(test_accuracy_rf)}")

from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
x = pd.read_csv("feature_VBL-VA001.csv", header=None)
y = pd.read_csv("label_VBL-VA001.csv", header=None)
y = pd.Series.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True
)

# KNN
knn = KNeighborsClassifier(n_neighbors=2)
out_knn = knn.fit(X_train, y_train)
print("KNN Accuracy on Train Data: {}".format(knn.score(X_train, y_train)))
print("KNN Accuracy on Test Data: {}".format(knn.score(X_test, y_test)))

# SVM Machine Learning
svm = SVC(C=86, kernel='rbf', class_weight='balanced', random_state=None)
out_svm = svm.fit(X_train, y_train)
print("SVM accuracy is {} on Train Dataset".format(svm.score(X_train, y_train)))
print("SVM accuracy is {} on Test Dataset".format(svm.score(X_test, y_test)))

# Naive Bayes
model = GaussianNB(var_smoothing=1e-11)
out_gnb = model.fit(X_train, y_train)
gnb_pred = model.predict(X_test)

print("NB accuracy is {} on Train Dataset".format(model.score(X_train, y_train)))
print("NB accuracy is {} on Test Dataset".format(model.score(X_test, y_test)))

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
out_rf = rf.fit(X_train, y_train)
print("Random Forest accuracy is {} on Train Dataset".format(rf.score(X_train, y_train)))
print("Random Forest accuracy is {} on Test Dataset".format(rf.score(X_test, y_test)))

# Class names
class_names = ['Normal', 'Misalignment', 'Unbalance', 'Bearing']

# Plot confusion matrix for each classifier
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
classifiers = [out_knn, out_svm, out_gnb, out_rf]

for i, ax in enumerate(axes.flatten()):
    if i < len(classifiers):
        classifier = classifiers[i]
        classifier_name = type(classifier).__name__
        disp = ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, ax=ax,
                                                     display_labels=class_names,
                                                     cmap='YlGn',
                                                     values_format='.2f',
                                                     xticks_rotation=45,
                                                     normalize='true',
                                                     colorbar=False)
        disp.ax_.set_title(f'Confusion Matrix - {classifier_name}')
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()

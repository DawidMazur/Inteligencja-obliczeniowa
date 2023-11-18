import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('diabetes.csv')

from sklearn.model_selection import train_test_split

(train_set, test_set) = train_test_split(df.values, train_size=0.7,
                                         random_state=2)

train_inputs = train_set[:, 0:8]
train_labels = train_set[:, 8]
test_inputs = test_set[:, 0:8]
test_labels = test_set[:, 8]

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(
    solver='lbfgs',
    activation='relu',
    hidden_layer_sizes=(6, 3),
    max_iter=800,
)

clf.fit(train_inputs, train_labels)


def accuracy(cm):
    diagonal_sum = cm.trace()
    sum_of_all = cm.sum()
    return diagonal_sum / sum_of_all


def check(clfs, test_inputs, code):
    pred = clfs.predict(test_inputs)
    cm = confusion_matrix(pred, code)
    print("Accuracy:", accuracy(cm))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=clf.classes_)
    disp.plot()
    plt.show()




check(clf, test_inputs, test_labels)

import pandas as pd
from sklearn.metrics import confusion_matrix

df = pd.read_csv('iris.csv')

from sklearn.model_selection import train_test_split

(train_set, test_set) = train_test_split(df.values, train_size=0.7,
                                         random_state=2)

train_inputs = train_set[:, 0:4]
train_labels = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_labels = test_set[:, 4]

from sklearn.neural_network import MLPClassifier

clf_2 = MLPClassifier(
    solver='lbfgs',
    hidden_layer_sizes=(2),
)
clf_2.fit(train_inputs, train_labels)

clf_3 = MLPClassifier(
    solver='lbfgs',
    hidden_layer_sizes=(3)
)
clf_3.fit(train_inputs, train_labels)

clf_3_3 = MLPClassifier(
    solver='lbfgs',
    hidden_layer_sizes=(3, 3)
)
clf_3_3.fit(train_inputs, train_labels)


def accuracy(cm):
    diagonal_sum = cm.trace()
    sum_of_all = cm.sum()
    return diagonal_sum / sum_of_all


def test(clf, test_inputs, code):
    pred = clf.predict(test_inputs)
    cm = confusion_matrix(pred, code)
    print("Accuracy:", accuracy(cm))


test(clf_2, test_inputs, test_labels)
test(clf_3, test_inputs, test_labels)
test(clf_3_3, test_inputs, test_labels)

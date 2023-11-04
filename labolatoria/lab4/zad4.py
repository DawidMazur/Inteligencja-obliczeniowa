import pandas as pd

df = pd.read_csv('iris.csv')

from sklearn.model_selection import train_test_split

(train_set, test_set) = train_test_split(df.values, train_size=0.7,
                                         random_state=2)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]


def knn(train_inputs, train_classes, test_inputs, k):
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_inputs, train_classes)

    return knn.predict(test_inputs)


def nb(train_inputs, train_classes, test_inputs):
    from sklearn.naive_bayes import GaussianNB

    model = GaussianNB()
    model.fit(train_inputs, train_classes)

    return model.predict(test_inputs)


def dd(train_inputs, train_classes, test_inputs):
    from sklearn.tree import DecisionTreeClassifier

    decision_tree = DecisionTreeClassifier(random_state=123)
    decision_tree = decision_tree.fit(train_inputs, train_classes)

    return decision_tree.predict(test_inputs)


def accuracy(test_classes, predict_classes):
    good_predictions = 0
    len = test_classes.shape[0]

    for i in range(len):
        if predict_classes[i] == test_classes[i]:
            good_predictions = good_predictions + 1

    return good_predictions / len * 100


knn3_predict = knn(train_inputs, train_classes, test_inputs, 3)
knn5_predict = knn(train_inputs, train_classes, test_inputs, 5)
knn11_predict = knn(train_inputs, train_classes, test_inputs, 11)
nb_predict = nb(train_inputs, train_classes, test_inputs)
dd_predict = dd(train_inputs, train_classes, test_inputs)

print("DD accuracy: ", accuracy(test_classes, dd_predict), "%")
print("KNN3 accuracy: ", accuracy(test_classes, knn3_predict), "%")
print("KNN5 accuracy: ", accuracy(test_classes, knn5_predict), "%")
print("KNN11 accuracy: ", accuracy(test_classes, knn11_predict), "%")
print("NB accuracy: ", accuracy(test_classes, nb_predict), "%")

# plot with accuracy
import matplotlib.pyplot as plt

plt.bar(["DD", "KNN3", "KNN5", "KNN11", "NB"],
        [accuracy(test_classes, dd_predict),
         accuracy(test_classes, knn3_predict),
         accuracy(test_classes, knn5_predict),
         accuracy(test_classes, knn11_predict),
         accuracy(test_classes, nb_predict)])
plt.show()

# print(test_result)
# print(test_classes)
# print(neigh.kneighbors_graph(test_inputs).toarray())

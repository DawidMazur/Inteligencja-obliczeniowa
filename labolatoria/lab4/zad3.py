import pandas as pd

df = pd.read_csv('iris.csv')

from sklearn.model_selection import train_test_split

(train_set, test_set) = train_test_split(df.values, train_size=0.7,
                                         random_state=125)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

decision_tree = DecisionTreeClassifier(random_state=123)
decision_tree = decision_tree.fit(train_inputs, train_classes)

result = export_text(decision_tree, feature_names=list(df.columns.values[0:4]))
print(result)

good_predictions = 0
len = test_set.shape[0]

for i in range(len):
    if decision_tree.predict([test_inputs[i]]) == test_classes[i]:
        good_predictions = good_predictions + 1

print(good_predictions)
print(good_predictions/len*100, "%")

# confusion matrix
from sklearn.metrics import confusion_matrix

y_pred = decision_tree.predict(test_inputs)
print(confusion_matrix(test_classes, y_pred))
# as plot
import seaborn as sn
import matplotlib.pyplot as plt

df_cm = pd.DataFrame(confusion_matrix(test_classes, y_pred),
                        index=["Setosa", "Versicolor", "Virginica"],
                        columns=["Setosa", "Versicolor", "Virginica"])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.show()


# 45
# 100.0 %

# jest lepiej
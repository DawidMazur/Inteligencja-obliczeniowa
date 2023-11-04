import pandas as pd

df = pd.read_csv('iris.csv')

from sklearn.model_selection import train_test_split

(train_set, test_set) = train_test_split(df.values, train_size=0.7,
                                         random_state=275520)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

print(train_set)

def classify_iris(sl, sw, pl, pw):
    if pw < 1.0:
        return ("Setosa")
    elif sl >= 4.9 and pw >= 1.4 and pl >= 4.9:
        return ("Virginica")
    else:
        return ("Versicolor")

def classify_iris_tab(x):
    return classify_iris(x[0], x[1], x[2], x[3])

good_predictions = 0
len = test_set.shape[0]

for i in range(len):
    if classify_iris_tab(test_inputs[i]) == test_classes[i]:
        good_predictions = good_predictions + 1
print(good_predictions)
print(good_predictions/len*100, "%")

# 44
# 97.77777777777777 %


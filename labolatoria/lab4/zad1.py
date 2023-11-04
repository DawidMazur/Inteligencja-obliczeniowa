import pandas as pd

df = pd.read_csv('iris.csv')

# print(df.values)
#
# #wszystkie wiersze, kolumna nr 0
# print(df.values[:, 0])
# #wiersze od 5 do 10, wszystkie kolumny
# print(df.values[5:11, :])
# #dane w komórce [1,4]
# print(df.values[1, 4])

from sklearn.model_selection import train_test_split

#podzial na zbior testowy (30%) i treningowy (70%), ziarno losowosci = 13
(train_set, test_set) = train_test_split(df.values, train_size=0.7,
random_state=13)

print(test_set)
print(test_set.shape[0])

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]


print(train_inputs)
print(train_classes)
print(test_inputs)
print(test_classes)

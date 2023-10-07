import math

w1 = [3, 8, 9, 10, 12]
w2 = [8, 7, 7, 5, 6]

wsum = [w1[i] + w2[i] for i in range(len(w1))]
print(wsum)

wmultiply = [w1[i] * w2[i] for i in range(len(w1))]
print(wmultiply)


# iloczyn skalarny
def dot_product(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])


print(dot_product(w1, w2))


# dlugosci euklidesowe
def norm(x):
    return math.sqrt(dot_product(x, x))


print(norm(w1))
print(norm(w2))


# wektor 50 losowych liczb z przedzialu [1, 100]
import random

w3 = [random.randint(1, 100) for _ in range(50)]
print(w3)

avg = sum(w3) / len(w3)
print(avg)
min = min(w3)
print(min)
max = max(w3)
print(max)
odchylenie_std = math.sqrt(sum([(x - avg) ** 2 for x in w3]) / len(w3))
print(odchylenie_std)

#normalizacja
w4 = [(x - min) / (max - min) for x in w3]

print(w4)


#standaryzacja
w5 = [(x - avg) / odchylenie_std for x in w3]

print(w5)


w5mean = sum(w5) / len(w5)
print(w5mean)
w5odchylenie_std = math.sqrt(sum([(x - w5mean) ** 2 for x in w5]) / len(w5))
print(w5odchylenie_std)
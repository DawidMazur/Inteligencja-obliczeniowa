import pandas as pd
import matplotlib.pyplot as plt

miasta = pd.read_csv('miasta.csv', sep=',')
print(miasta)

# b)
miasta.loc[10] = [2010, 460, 555, 405]
print(miasta)

plt.plot(miasta['Rok'], miasta['Gdansk'], marker='o', color='red' )
plt.plot(miasta['Rok'], miasta['Poznan'], marker='o', color='blue' )
plt.plot(miasta['Rok'], miasta['Szczecin'], marker='o', color='green' )
plt.title('Ludność w miascie Gdańsk')
plt.xlabel('Rok')
plt.ylabel('Liczba ludności [w tyś.]')
plt.legend(['Gdańsk', 'Poznań', 'Szczecin'])
plt.show()
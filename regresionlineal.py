import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize']=(16,9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#configurar Pandas para que muestre todas las columnas sin truncarlas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)  
pd.set_option('display.max_colwidth', None)

#cargar datos de entrada
ruta = r"C:\Users\isabe\OneDrive - Universidad Autonoma de Nuevo León\Inteligencia Artificial\Marzo\Regresion lineal multiple\articulos_ml.csv"

data = pd.read_csv(ruta)
# Muestra las dimensiones del DataFrame
print("Forma de los datos:", data.shape)  
#Muestra los primeros 5 registros 
print(data.head())
#Estadísticas de los datos 
print(data.describe())
#Visualización de las caracteristicas de entrada
data.drop(['Title','url','Elapsed days'],axis=1).hist()
plt.show()

#filtrar datos
filtered_data = data[(data['Word count'] <= 3500) & (data['# Shares']<=80000)]

colores=['orange','blue']
tamanios=[30,60]

f1=filtered_data['Word count'].values
f2=filtered_data['# Shares'].values

#Pintar en colores los puntis por debajo y encima de la media de Catidad de palabras
asignar=[]
for index, row in filtered_data.iterrows():
    if(row['Word count']>1808):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
plt.scatter(f1,f2,c=asignar,s=tamanios[0])
plt.show()

#regresion lineal con python y sklearn
dataX = filtered_data[['Word count']]
X_train = np.array(dataX)
y_train = filtered_data['# Shares'].values

#Creacion del objeto regresión linear
regresion= linear_model.LinearRegression()

#Entrenamiento del modelo
regresion.fit(X_train,y_train)

#Predicciones del modelo
y_pred= regresion.predict(X_train)
#Coeficientes obtenidos 
print('Coeficientes: \n', regresion.coef_)
#Valor donde corta el eje Y (en X=0)
print('Terminos independientes: \n',regresion.intercept_)
#Error cuadrado medio
print('Cuadrado medio del error: %.2f' %mean_squared_error(y_train, y_pred))
#Valor de la varianza
print('Valor de la varianza: %.2f' %r2_score(y_train, y_pred))

y_dosmil= regresion.predict([[2000]])
print("predicción de “Shares” para un artículo de 2000 palabras",int(y_dosmil))
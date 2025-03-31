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
#cargar datos de entrada
ruta = r"C:\Users\isabe\OneDrive - Universidad Autonoma de Nuevo León\Inteligencia Artificial\Marzo\Regresion lineal multiple\articulos_ml.csv"
data = pd.read_csv(ruta)
#filtrar datos 
filtered_data = data[(data['Word count'] <= 3500) & (data['# Shares']<=80000)]
#Variable suma  de los enlaces, comentarios e imágenes
suma=(filtered_data['# of Links']+filtered_data['# of comments'].fillna(0)+filtered_data['# Images video'])

dataX2 = pd.DataFrame()
dataX2["Word count"]= filtered_data["Word count"]
dataX2["suma"]=suma
XY_train=np.array(dataX2)
z_train=filtered_data['# Shares'].values

#nuevo objeto de regresion lineal
regr2 = linear_model.LinearRegression()

#Entrenar el modelo con 2 dimensiones
regr2.fit(XY_train,z_train)
z_pred= regr2.predict(XY_train)

#Coeficientes obtenidos 
print('Coeficientes:', regr2.coef_)

#Error cuadrado medio
print('Cuadrado medio del error: %.2f' %mean_squared_error(z_train, z_pred))
#Valor de la varianza
print('Valor de la varianza: %.2f' %r2_score(z_train, z_pred))

#Visualizacion de un plano en 3 dimensiones
fig = plt.figure()
ax = Axes3D(fig)

#Creacion de una malla, sobre la cual se graficara el plano
xx, yy = np.meshgrid(np.linspace(0,3500, num=10),np.linspace(0,60,num=10))
#calcular los puntos del plano4
nuevoX = (regr2.coef_[0] * xx)
nuevoY = (regr2.coef_[1] * yy)

# calculamos los correspondientes valores para z. Debemos sumar el punto de intercepción
z = (nuevoX + nuevoY + regr2.intercept_)
# Graficamos el plano
ax.plot_surface(xx, yy, z, alpha=0.2, cmap='hot')

# Graficamos en azul los puntos en 3D
ax.scatter(XY_train[:, 0], XY_train[:, 1], z_train, c='blue',s=30)
 
# Graficamos en rojo, los puntos que
ax.scatter(XY_train[:, 0], XY_train[:, 1], z_pred, c='red',s=40)
 
# con esto situamos la "camara" con la que visualizamos
ax.view_init(elev=30., azim=65)
 
ax.set_xlabel('Cantidad de Palabras')
ax.set_ylabel('Cantidad de Enlaces,Comentarios e Imagenes')
ax.set_zlabel('Compartido en Redes')
ax.set_title('Regresión Lineal con Múltiples Variables')
plt.show()

# Si quiero predecir cuántos "Shares" voy a obtener por un artículo con:
# 2000 palabras y con enlaces: 10, comentarios: 4, imagenes: 6
# según nuestro modelo, hacemos:
z_Dosmil = regr2.predict([[2000, 10+4+6]])
print('Cantidad de shares de 2000 palabras, 10 enlaces, 4 comentarios y 6 imágenes: ', int(z_Dosmil))
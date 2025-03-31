import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


dataframe = pd.read_csv(r"usuarios_win_mac_lin.csv")
print(dataframe.head())
print(dataframe.describe())

#Cuantos usuarios hay de cada tipo
print (dataframe.groupby('clase').size())

dataframe.drop(['clase'],axis=1).hist()
plt.show()

sb.pairplot(dataframe.dropna(), hue='clase', size=4, vars=["duracion","paginas","acciones","valor"],kind='reg')
plt.show()

#excluir la columna clase
X=np.array(dataframe.drop(['clase'], axis=1))
#agregar a la columna clase en una nueva variable
y= np.array(dataframe['clase'])
#comprobar la dimension de la matriz
X.shape

#creacion del modelo de regresion logistica
model = linear_model.LogisticRegression()
model.fit(X,y)

predictions = model.predict(X)
print(predictions[0:5])

print(model.score(X, y))

#validacion del modelo
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,y,test_size=validation_size, random_state=seed)

name='Logistic Regression'
kfold = model_selection.KFold(n_splits=10, shuffle=True ,random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)

predictions = model.predict(X_validation)
print(accuracy_score(Y_validation,predictions))

#reporte de resultados
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation,predictions))

#clasificacion de nuevos valores
X_new= pd.DataFrame({'duracion':[10], 'paginas': [3], 'acciones':[5], 'valor':[9]})
print(model.predict(X_new))

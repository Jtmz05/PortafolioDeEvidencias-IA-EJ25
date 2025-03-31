import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from graphviz import Source

artists_billboard = pd.read_csv(r"artists_billboard_fix3.csv")
print(artists_billboard.shape)
print(artists_billboard.head())
print(artists_billboard.groupby('top').size())

sb.catplot(x="artist_type", data=artists_billboard, kind="count", palette="viridis")
plt.show()

sb.catplot(x="mood", data=artists_billboard, kind="count", aspect=3, palette="coolwarm")
plt.show()

sb.catplot(x='tempo', data=artists_billboard,hue='top', kind="count", palette="coolwarm")
plt.show()

sb.catplot(x='genre', data=artists_billboard, kind="count", aspect=3, palette='coolwarm')
plt.show()

sb.catplot(x='anioNacimiento', data=artists_billboard, kind="count", aspect=3, palette='coolwarm')
plt.show()

f1 = artists_billboard['chart_date'].values
f2 = artists_billboard['durationSeg'].values

colores = ['orange', 'blue']
tamanios=[60,40]
asignar=[]
asignar2=[]
for index, row in artists_billboard.iterrows():
    asignar.append(colores[row['top']])
    asignar2.append(tamanios[row['top']])

plt.scatter(f1, f2, c=asignar, s=asignar2)
plt.axis([20030101, 20160101, 0, 600])
plt.show()

def edad_fix(anio):
   if anio==0:
     return None
   return anio

artists_billboard['anioNacimiento']=artists_billboard.apply(lambda x:edad_fix(x['anioNacimiento']),axis=1)

def calcula_edad(anio,cuando):
   cad=str(cuando)
   momento=cad[:4]
   if anio==0.0:
     return None
   return int(momento)-anio
   
artists_billboard['edad_en_billboard']=artists_billboard.apply(lambda x: calcula_edad(x['anioNacimiento'],x['chart_date']), axis=1) 

age_avg = artists_billboard['edad_en_billboard'].mean()
age_std = artists_billboard['edad_en_billboard'].std()
age_null_count = artists_billboard['edad_en_billboard'].isnull().sum()
age_null_random_list=np.random.randint(age_avg-age_std,age_avg+age_std,size=age_null_count)

conValoresNulos = np.isnan(artists_billboard['edad_en_billboard'])
artists_billboard.loc[np.isnan(artists_billboard['edad_en_billboard']),'edad_en_billboard']= age_null_random_list
artists_billboard['edad_en_billboard']= artists_billboard['edad_en_billboard'].astype(int)

print("Edad promedio: " + str(age_avg))
print("Desviacion estandar de la edad: "+ str(age_std))
print("Intervalo para asignar edad aleatoria: "+ str(int(age_avg-age_std))+" a "+ str(int(age_avg+age_std)))

f1= artists_billboard['edad_en_billboard'].values
f2 = artists_billboard.index

colores = ['orange', 'blue', 'green']
asignar=[]

for index, row in artists_billboard.iterrows():
   if(conValoresNulos[index]):
      asignar.append(colores[2])
   else:
      asignar.append(colores[row['top']])

plt.scatter(f1, f2, c=asignar, s=30)
plt.axis([15,50,0,650])
plt.show()

artists_billboard['moodEncoded']= artists_billboard['mood'].map({'Energizing':6,
            'Empowering':6,
            'Cool': 5,
            'Yearning':4,
            'Excited': 5,
            'Defiant':3,
            'Sensual':2,
            'Gritty':3,
            'Sophisticated':4,
            'Aggressive':4,
            'Fiery':4,
            'Urgent':3,
            'Rowdy':4,
            'Sentimental': 4,
            'Easygoing': 1,
            'Melancholy': 4,
            'Romantic': 2,
            'Peaceful': 1,
            'Brooding': 4,
            'Upbeat': 5,
            'Stirring': 5,
            'Lively': 5,
            'Other': 0,
            '': 0}).astype(int)
artists_billboard['tempoEncoded']=artists_billboard['tempo'].map({'Fast Tempo':0, 'Medium Tempo':2,'Slow Tempo':1,'':0}).astype(int)
artists_billboard['genreEncoded']=artists_billboard['genre'].map({'Urban':4,
            'Pop':3,
            'Traditional': 2,
            'Alternative & Punk': 1,
            'Electronica': 1,
            'Rock': 1,
            'Soundtrack': 0,
            'Jazz': 0,
            'Other': 0,
            '': 0}
            ).astype(int)

# Mapeo de tipos de artistas
artists_billboard['artist_typeEncoded'] = artists_billboard['artist_type'].map({
    'Female': 2,
    'Male': 3,
    'Mixed': 1,
    '': 0
}).astype(int)

# Mapeo de edad en la que llegaron al Billboard
artists_billboard.loc[artists_billboard['edad_en_billboard'] <= 21, 'edadEncoded'] = 0
artists_billboard.loc[(artists_billboard['edad_en_billboard'] > 21) & 
                      (artists_billboard['edad_en_billboard'] <= 26), 'edadEncoded'] = 1
artists_billboard.loc[(artists_billboard['edad_en_billboard'] > 26) & 
                      (artists_billboard['edad_en_billboard'] <= 30), 'edadEncoded'] = 2
artists_billboard.loc[(artists_billboard['edad_en_billboard'] > 30) & 
                      (artists_billboard['edad_en_billboard'] <= 40), 'edadEncoded'] = 3
artists_billboard.loc[artists_billboard['edad_en_billboard'] > 40, 'edadEncoded'] = 4

# Mapeo de duración de la canción
artists_billboard.loc[artists_billboard['durationSeg'] <= 150, 'durationEncoded'] = 0
artists_billboard.loc[(artists_billboard['durationSeg'] > 150) & 
                      (artists_billboard['durationSeg'] <= 180), 'durationEncoded'] = 1
artists_billboard.loc[(artists_billboard['durationSeg'] > 180) & 
                      (artists_billboard['durationSeg'] <= 210), 'durationEncoded'] = 2
artists_billboard.loc[(artists_billboard['durationSeg'] > 210) & 
                      (artists_billboard['durationSeg'] <= 240), 'durationEncoded'] = 3
artists_billboard.loc[(artists_billboard['durationSeg'] > 240) & 
                      (artists_billboard['durationSeg'] <= 270), 'durationEncoded'] = 4
artists_billboard.loc[(artists_billboard['durationSeg'] > 270) & 
                      (artists_billboard['durationSeg'] <= 300), 'durationEncoded'] = 5
artists_billboard.loc[artists_billboard['durationSeg'] > 300, 'durationEncoded'] = 6

drop_elements = [
    'id', 'title', 'artist', 'mood', 'tempo', 'genre', 
    'artist_type', 'chart_date', 'anioNacimiento', 
    'durationSeg', 'edad_en_billboard'
]

artists_encoded = artists_billboard.drop(drop_elements, axis=1)
print(artists_encoded[['moodEncoded', 'top']].groupby(['moodEncoded'], as_index=False).agg(['mean', 'count', 'sum']))
print(artists_encoded[['artist_typeEncoded', 'top']].groupby(['artist_typeEncoded'], as_index=False).agg(['mean', 'count', 'sum']))
print(artists_encoded[['genreEncoded', 'top']].groupby(['genreEncoded'], as_index=False).agg(['mean', 'count', 'sum']))
print(artists_encoded[['tempoEncoded', 'top']].groupby(['tempoEncoded'], as_index=False).agg(['mean', 'count', 'sum']))
print(artists_encoded[['durationEncoded', 'top']].groupby(['durationEncoded'], as_index=False).agg(['mean', 'count', 'sum']))
print(artists_encoded[['edadEncoded', 'top']].groupby(['edadEncoded'], as_index=False).agg(['mean', 'count', 'sum']))

#Creación del arbol 
cv = KFold(n_splits=10)  # Número deseado de "folds" que haremos
accuracies = list()
max_attributes = len(list(artists_encoded))
depth_range = range(1, max_attributes + 1)

# Testearemos la profundidad de 1 a cantidad de atributos + 1
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(
        criterion='entropy',
        min_samples_split=20,
        min_samples_leaf=5,
        max_depth=depth,
        class_weight={1: 3.5}
    )
    
    for train_fold, valid_fold in cv.split(artists_encoded):
        f_train = artists_encoded.loc[train_fold]
        f_valid = artists_encoded.loc[valid_fold]
        
        model = tree_model.fit(X=f_train.drop(['top'], axis=1), y=f_train["top"])
        valid_acc = model.score(X=f_valid.drop(['top'], axis=1), y=f_valid["top"])  # calculamos la precisión con el segmento de validación
        fold_accuracy.append(valid_acc)
    
    avg = sum(fold_accuracy) / len(fold_accuracy)
    accuracies.append(avg)

# Mostramos los resultados obtenidos
df = pd.DataFrame({"MaxDepth": depth_range, "AverageAccuracy": accuracies})
df = df[["MaxDepth", "AverageAccuracy"]]
print(df.to_string(index=False))

# Crear arrays de entrenamiento y las etiquetas que indican si llegó a top o no
y_train = artists_encoded['top']
x_train = artists_encoded.drop(['top'], axis=1).values

# Crear Árbol de decisión con profundidad = 4
decision_tree = tree.DecisionTreeClassifier(
    criterion='entropy',
    min_samples_split=20,
    min_samples_leaf=5,
    max_depth=4,
    class_weight={1: 3.5}
)
decision_tree.fit(x_train, y_train)

# Exportar el modelo a archivo .dot
with open(r"tree1.dot", 'w') as f:
    tree.export_graphviz(
        decision_tree,
        out_file=f,
        max_depth=7,
        impurity=True,
        feature_names=list(artists_encoded.drop(['top'], axis=1)),
        class_names=['No', 'N1Billboard'],
        rounded=True,
        filled=True
    )

# Crear y renderizar el gráfico con Graphviz
source = Source.from_file("tree1.dot")
source.render('tree1', format='png', view=True)

# Calcular la precisión del modelo de árbol de decisión en el conjunto de entrenamiento
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
print(acc_decision_tree)

# Predicción para el artista Camila Cabello featuring Young Thug
x_test = pd.DataFrame(columns=('top','moodEncoded', 'tempoEncoded', 'genreEncoded', 'artist_typeEncoded', 'edadEncoded', 'durationEncoded'))
x_test.loc[0] = (1, 5, 2, 4, 1, 0, 3)  # Valores de características para el test
y_pred = decision_tree.predict(x_test.drop(['top'], axis=1))
print("Predicción: " + str(y_pred))

y_proba = decision_tree.predict_proba(x_test.drop(['top'], axis=1))
print("Probabilidad de Acierto: " + str(round(y_proba[0][y_pred[0]] * 100, 2)) + "%")  # Usamos y_pred[0] como índice

# Predicción para el artista Imagine Dragons
x_test.loc[0] = (0, 4, 2, 1, 3, 2, 3)  # Valores de características para el test
y_pred = decision_tree.predict(x_test.drop(['top'], axis=1))
print("Predicción: " + str(y_pred))
y_proba = decision_tree.predict_proba(x_test.drop(['top'], axis=1))
print("Probabilidad de Acierto: " + str(round(y_proba[0][y_pred[0]] * 100, 2)) + "%")  # Usamos y_pred[0] como índice


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

dataframe= pd.read_csv(r"reviews_sentiment.csv", sep=';')
print(dataframe.head(10))

print(dataframe.describe())

dataframe.hist()
plt.show()

print(dataframe.groupby('Star Rating').size())
sb.catplot(x='Star Rating', data=dataframe, kind="count", aspect=3,palette="Set2")
plt.show()

sb.catplot(x='wordcount',data=dataframe, kind="count", aspect=3, palette="Set3")
plt.show()

X = dataframe[['wordcount','sentimentValue']].values
y = dataframe['Star Rating'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_neighbors = 7
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)

print('Accuracy of K-NN classifier on training set {:.2f}'.format(knn.score(X_train,y_train)))
print('Accuracy of K-NN classifier on test set {:.2f}'.format(knn.score(X_test,y_test)))

pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

h=.02
weights = 'distance' 

cmap_light = ListedColormap(['#FFAAAA', '#ffcc99', '#ffffb3','#b3ffff','#c2f0c2'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933','#FFFF00','#00ffff','#00FF00'])

clf = KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(X,y)

x_min, x_max =X[:,0].min()-1,X[:,0].max()+1
y_min, y_max =X[:,1].min()-1,X[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max,h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z= Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(X[:,0],X[:,1],c=y, cmap=cmap_bold, edgecolor='k',s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

patch0 = mpatches.Patch(color='#FF0000')
patch1 = mpatches.Patch(color='#ff9933')
patch2 = mpatches.Patch(color='#FFFF00')
patch3 = mpatches.Patch(color='#00ffff')
patch4 = mpatches.Patch(color='#00FF00')

plt.legend(handles=[patch0, patch1, patch2, patch3, patch4])
plt.title("5-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))
plt.show()

k_range = range(1,20)
scores= []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range,scores)
plt.xticks([0,5,10,15,20])
plt.show()

print(clf.predict([[5,1.0]]))
print(clf.predict_proba([[20,0.0]]))

import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


dataframe= pd.read_csv(r"creditcard.csv")
print(dataframe.head())

count_classes =pd.value_counts(dataframe['Class'],sort=True)
count_classes.plot(kind='bar',rot=0)

LABELS = ["No Fraud", "Fraud"]
plt.xticks(range(2),LABELS)

plt.title("Frequency by observation number")
plt.xlabel("Class")
plt.ylabel("Number of OBservations")
plt.show()

#definir etiquets y features
y = dataframe['Class']
X = dataframe.drop('Class', axis=1)

#sets de entrenamiento y test
X_train, X_test, y_train, y_test=train_test_split(X, y, train_size=0.7)

model = RandomForestClassifier(n_estimators=100,
                               bootstrap=True, verbose=2,
                               max_features='sqrt')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Reporte de clasificacion")
print(classification_report(y_test,predictions))

conf_matrix= confusion_matrix(y_test, predictions)
plt.figure(figsize=(12,12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel("True class")
plt.xlabel("Predict class")
plt.show()
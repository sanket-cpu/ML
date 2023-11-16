import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score , precision_score , recall_score
from sklearn.tree import DecisionTreeClassifier

zoo_data = pd.read_csv("zoo_data.csv")

X = zoo_data.iloc[: , :-1]
y = zoo_data.iloc[: , -1]

X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.3,train_size=0.7)

dt_model = DecisionTreeClassifier()

dt_model.fit(X_train,y_train)
y_pred = dt_model.predict(X_test)

conf = confusion_matrix(y_test , y_pred)
print("Confusion Matrix is\n" , conf)

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred, average=None)
recall = recall_score(y_test,y_pred, average=None)

print("Accuracy: " ,accuracy)
print("Precision: " ,precision)
print("Recall: " ,recall)

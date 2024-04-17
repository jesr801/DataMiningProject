import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import time

dataFinal = pd.read_csv('New_Police_Incidents.csv')

x = pd.DataFrame(dataFinal.Neighborhood)
y = pd.DataFrame(dataFinal.UCR_1_Code)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.5, random_state=1)


re = LinearRegression()
re.fit(x_train,y_train)
y_pre_re = re.predict(x_test)
print("Y_Pre:\n",y_pre_re)
print("\nScore X & Y Test:\n", re.score(x_test,y_test))
print('\n----------------------------------\n')

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(5, 2), random_state=1)
# clf = clf.fit(x_train, y_train.values.ravel())
#
# clf.predict(x_test)
# print("\nMLP Score:\n", clf.score(x_test,y_test))
# print('\n----------------------------------\n')

feat_cols = ['UCR_1_Code', 'UCR_2_Code']

X_tree = dataFinal[feat_cols]
y_tree = dataFinal.Neighborhood

X_train, X_test, y_train, y_test = train_test_split(X_tree, y_tree, test_size=0.3, random_state=1)

tree_clf = DecisionTreeClassifier()

tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
import pandas
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import time
df = pandas.read_csv("New_Police_IncidentsV1.csv")




x = pandas.DataFrame(df.Neighborhood)
y = pandas.DataFrame(df.UCR_1_Code)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.5, random_state=1)


re = LinearRegression()
re.fit(x_train,y_train)
y_pre = re.predict(x_test)
print("Y_Pre:\n",y_pre)

print("\nLinear Regression Score:\n",re.score(x_test,y_test))

# x_train = np.ravel(x_train)
# y_train = np.ravel(y_train)
# x_test = np.ravel(x_test)
# y_test = np.ravel(y_test)


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf = clf.fit(x_train, y_train.values.ravel())

clf.predict(x_test)
print("\nMLP Score:\n", clf.score(x_test,y_test))
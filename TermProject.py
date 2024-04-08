import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
df = pandas.read_csv("New_Police_IncidentsV1.csv")


# x = pandas.DataFrame(df.Neighborhood)
# y = pandas.DataFrame(df.UCR_1_Code)

# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)


# re = LinearRegression()
# re.fit(x_train,y_train)
# y_pre = re.predict(x_test)
# print("Y_Pre:\n",y_pre)

# print("\nScore X & Y Test:\n",re.score(x_test,y_test))

# Load the diabetes dataset
X = df.Neighborhood
y = df.UCR_1_Code
n_samples = 20

X = X[:, np.newaxis, 2][:n_samples]
y = y[:n_samples]
p = X.argsort(axis=0)
X = X[p].reshape((n_samples, 1))
y = y[p]

# Create equal weights and then augment the last 2 ones
sample_weight = np.ones(n_samples) * 20
sample_weight[-2:] *= 30

plt.scatter(X, y, s=sample_weight, c='grey', edgecolor='black')

# The unweighted model
regr = LinearRegression()
regr.fit(X, y)
plt.plot(X, regr.predict(X), color='blue', linewidth=3, label='Unweighted model')

# The weighted model
regr = LinearRegression()
regr.fit(X, y, sample_weight)
plt.plot(X, regr.predict(X), color='red', linewidth=3, label='Weighted model')

# The weighted model - scaled weights
regr = LinearRegression()
sample_weight = sample_weight / sample_weight.max()
regr.fit(X, y, sample_weight)
plt.plot(X, regr.predict(X), color='yellow', linewidth=2, label='Weighted model - scaled', linestyle='dashed')
plt.xticks(());plt.yticks(());plt.legend()
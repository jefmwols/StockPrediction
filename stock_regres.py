import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score



from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

date_parser = pd.to_datetime
df = pd.read_csv("AAPL.csv", parse_dates=['Date'])
print(df.head())

X_key = 'Open'
y_key = 'Adj Close'

X = df[X_key].values.reshape(-1,1)
y = df[y_key].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)

print("\n***\nLinear")
print(clfreg.intercept_)
print(clfreg.coef_)
y_pred1 = clfreg.predict(X_test)
df_t = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred1.flatten()})
print(df_t)

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)
ridge2 = clfpoly2.named_steps['ridge']
print("\n***\nQuadratic 2")
print(ridge2.coef_)
y_pred2 = clfpoly2.predict(X_test)
df_t2 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred2.flatten()})
print(df_t2)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)
ridge3 = clfpoly2.named_steps['ridge']
print("\n***\nQuadratic 3")
print(ridge3.coef_)
y_pred3 = clfpoly3.predict(X_test)
df_t3 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred3.flatten()})
print(df_t3)



plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred1, color='red', linewidth=2)
plt.plot(X_test, y_pred2, color='blue', linewidth=2)
plt.plot(X_test, y_pred3, color='green', linewidth=2)
plt.show()
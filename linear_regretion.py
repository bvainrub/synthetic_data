import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Пример 1
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([5, 4, 3, 2, 1])
model = LinearRegression().fit(X, y)
print("Пример 1:", model.coef_, model.intercept_)

# Пример 2
y = np.array([2, 4, 6, 8, 10])
model = LinearRegression().fit(X, y)
print("Пример 2:", model.predict([[6]]))

# Пример 3
X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
y = np.array([2, 4, 6, 8])
model = LinearRegression().fit(X, y)
print("Пример 3:", model.coef_, model.intercept_)

# Пример 4
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)
print("Пример 4:", model.coef_)

# Пример 5
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1])
clf = LogisticRegression().fit(X, y)
print("Пример 5:", clf.predict([[2.5]]))

# Пример 6
y = np.array([1, 2, 3, 4, 10])
ridge = Ridge(alpha=1.0).fit(X, y)
print("Пример 6:", ridge.coef_)

# Пример 7
lasso = Lasso(alpha=0.1).fit(X, y)
print("Пример 7:", lasso.coef_)

# Пример 8
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)
print("Пример 8:", elastic.coef_)

# Пример 9
df = pd.DataFrame({
    "категория": ["A", "B", "A", "B"],
    "x": [1, 2, 3, 4],
    "y": [2, 4, 6, 8]
})
df = pd.get_dummies(df, columns=["категория"], drop_first=True)
model = LinearRegression().fit(df[["x", "категория_B"]], df["y"])
print("Пример 9:", model.coef_)

# Пример 10
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
print("Пример 10:", r2_score(y, y_pred))

# Пример 11
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LinearRegression().fit(X_train, y_train)
print("Пример 11:", model.score(X_test, y_test))

# Пример 12
scores = cross_val_score(LinearRegression(), X, y, cv=4)
print("Пример 12:", scores)

# Пример 13
data = load_diabetes()
X, y = data.data, data.target
model = LinearRegression().fit(X, y)
print("Пример 13:", model.score(X, y))

# Пример 14
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])
tree = DecisionTreeRegressor(max_depth=2).fit(X, y)
print("Пример 14:", tree.predict([[6]]))

# Пример 15
forest = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
print("Пример 15:", forest.predict([[6]]))

# Пример 16
gbr = GradientBoostingRegressor().fit(X, y)
print("Пример 16:", gbr.predict([[6]]))

# Пример 17
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 3 * X.squeeze() + 7 + np.random.randn(50) * 2
model = LinearRegression().fit(X, y)
print("Пример 17:", model.coef_, model.intercept_)

# Пример 18
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)
print("Пример 18:", model.coef_)

# Пример 19
y_pred = model.predict(X_poly)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
print("Пример 19:", mae, mse, rmse)

# Пример 20
time = np.arange(1, 21).reshape(-1, 1)
sales = np.array([5, 7, 9, 12, 15, 18, 21, 25, 28, 30,
                  33, 36, 39, 41, 44, 47, 50, 52, 55, 58])
model = LinearRegression().fit(time, sales)
print("Пример 20:", model.coef_, model.intercept_)

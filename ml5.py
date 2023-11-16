import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("C:/Users/omkar/Downloads/uber.csv", encoding='latin1')

df = df.drop(['Unnamed: 0', 'key'], axis=1)

df = df.dropna(axis=0)

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

print(df.dtypes)

df = df.assign(
hour = df.pickup_datetime.dt.hour,
day = df.pickup_datetime.dt.day,
month = df.pickup_datetime.dt.month,
year = df.pickup_datetime.dt.year,
dayofweek = df.pickup_datetime.dt.dayofweek,
)
df = df.drop("pickup_datetime", axis=1)
print(df.shape)
x = df.drop("fare_amount", axis=1)
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

"////////////////Linear Reg/////////////////"
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae=mean_absolute_error(y_test, y_pred)
print("MAE",mae)
print("MSE",mean_squared_error(y_test, y_pred))
print("RMAE",np.sqrt(mean_squared_error(y_test, y_pred)))

"////////////////Random Forest/////////////////"

model = RandomForestRegressor(n_estimators=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("MAE:",mean_absolute_error(y_test, y_pred))
print("MSE:",mean_squared_error(y_test, y_pred))
print("RMSE:",np.sqrt(mean_squared_error(y_test, y_pred)))
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

from preprocessing import load_and_split

x_train, x_test, y_train, y_test = load_and_split()

model = LinearRegression()
model.fit(x_train, y_train)


Permissions =model.predict(x_test)

mse =mean_squared_error(y_test, Permissions)
print(f'Mean Squared Error: {mse}')

joblib.dump(model, 'linear_regression_model.pkl')

print("Model saved successfully")
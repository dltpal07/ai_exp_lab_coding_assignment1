from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

class Model():
    def __init__(self):
        self.linear_regression_estimator = LinearRegression()

    def train(self, x_train, y_train):
        self.linear_regression_estimator.fit(x_train, y_train)

    def predict(self, x_val):
        pred_y = self.linear_regression_estimator.predict(x_val)
        return pred_y

    def rmse(self, pred_y, true_y):
        return math.sqrt(mean_squared_error(true_y, pred_y))
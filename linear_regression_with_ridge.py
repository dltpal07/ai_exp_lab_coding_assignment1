from sklearn.linear_model import Ridge
import math
from sklearn.metrics import mean_squared_error


class Model():
    def __init__(self):
        self.ridge_estimator = Ridge(alpha=1.0)

    def train(self, x_train, y_train):
        self.ridge_estimator.fit(x_train, y_train)

    def predict(self, x_val):
        pred_y = self.ridge_estimator.predict(x_val)
        return pred_y

    def rmse(self, pred_y, true_y):
        return math.sqrt(mean_squared_error(true_y, pred_y))
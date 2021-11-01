from sklearn.linear_model import Lasso
import math
from sklearn.metrics import mean_squared_error


class Model():
    def __init__(self):
        self.lasso_estimator = Lasso(alpha=1.0, max_iter=1000, tol=0.0001)

    def train(self, x_train, y_train):
        self.lasso_estimator.fit(x_train, y_train)

    def predict(self, x_val):
        pred_y = self.lasso_estimator.predict(x_val)
        return pred_y

    def rmse(self, pred_y, true_y):
        return math.sqrt(mean_squared_error(true_y, pred_y))
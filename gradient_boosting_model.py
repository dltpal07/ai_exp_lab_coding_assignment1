from sklearn import ensemble
import math
from sklearn.metrics import mean_squared_error


class Model():
    def __init__(self):
        params = {
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_split": 5,
            "learning_rate": 0.01
        }
        self.gbm_reg = ensemble.GradientBoostingRegressor(**params)

    def train(self, x_train, y_train):
        print('wait...(It might take a long time)')
        self.gbm_reg.fit(x_train, y_train)

    def predict(self, x_val):
        pred_y = self.gbm_reg.predict(x_val)
        return pred_y

    def rmse(self, pred_y, true_y):
        return math.sqrt(mean_squared_error(true_y, pred_y))
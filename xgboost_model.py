import xgboost
import math
from sklearn.metrics import mean_squared_error


class Model():
    def __init__(self):
        self.xgb_model = xgboost.XGBRegressor(n_estimators=2000, learning_rate=0.02, gamma=0, subsample=0.75,
                                     colsample_bytree=1, max_depth=6)

    def train(self, x_train, y_train):
        self.xgb_model.fit(x_train, y_train)

    def predict(self, x_val):
        pred_y = self.xgb_model.predict(x_val)
        return pred_y

    def rmse(self, pred_y, true_y):
        return math.sqrt(mean_squared_error(true_y, pred_y))
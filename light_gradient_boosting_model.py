import lightgbm as lgb
import math
from sklearn.metrics import mean_squared_error


class Model():
    def __init__(self):
        self.params = {'learning_rate': 0.0001,
                  'max_depth': -1,
                  'objective': 'regression',
                  'metric': 'rmse',
                  'is_training_metric': True,
                  'num_leaves': 50,
                  'feature_fraction': 0.9,
                  'bagging_fraction': 0.7,
                  'bagging_freq': 5,
                  'seed': 2020,
                  'max_bin': 32,
                  'num_iterations': 100000}
        self.lgb_model = None

    def train(self, x_train, y_train):
        train_ds = lgb.Dataset(x_train, label=y_train)
        self.lgb_model = lgb.train(self.params, train_ds, 1000, train_ds, verbose_eval=100, early_stopping_rounds=100)

    def predict(self, x_val):
        pred_y = self.lgb_model.predict(x_val)
        return pred_y

    def rmse(self, pred_y, true_y):
        return math.sqrt(mean_squared_error(true_y, pred_y))
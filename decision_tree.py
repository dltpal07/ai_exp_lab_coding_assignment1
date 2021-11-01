from sklearn.tree import DecisionTreeRegressor
import math
from sklearn.metrics import mean_squared_error


class Model():
    def __init__(self):
        self.decision_tree_regr = DecisionTreeRegressor(max_depth=25)

    def train(self, x_train, y_train):
        self.decision_tree_regr.fit(x_train, y_train)

    def predict(self, x_val):
        y_pred = self.decision_tree_regr.predict(x_val)
        return y_pred

    def rmse(self, pred_y, true_y):
        return math.sqrt(mean_squared_error(true_y, pred_y))

from sklearn import metrics
import math
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
import xgboost
import lightgbm as lgb
import csv
import logging
from tqdm import tqdm
from argparse import Namespace
import argparse
import sys

log = logging.getLogger(__name__)
log.handlers = []
log.setLevel(logging.INFO)

def load_data():
    train_data = pd.read_csv('data/price_data_tr.csv')
    val_data = pd.read_csv('data/price_data_val.csv')
    test_data = pd.read_csv('data/price_data_ts.csv')
    return train_data, val_data, test_data


def linear_model(x_train, y_train, x_val, y_val, x_test, num_epochs):
    x_train_tensor = torch.from_numpy(pd.get_dummies(x_train).values).float()
    y_train_tensor = torch.from_numpy(y_train.values).float().unsqueeze(1)
    x_val_tensor = torch.from_numpy(pd.get_dummies(x_val).values).float()
    y_val_tensor = torch.from_numpy(y_val.values).float().unsqueeze(1)
    x_test_tensor = torch.from_numpy(pd.get_dummies(x_test).values).float()
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    batch_size = len(x_train_tensor)
    train_dl = DataLoader(train_data, batch_size, shuffle=False)

    linear_model = net = torch.nn.Linear(18, 1)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=0.01)
    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            for step, (xb, yb) in enumerate(train_dl):
                pred = linear_model(xb)
                loss = loss_fn(pred, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pbar.update(1)
            pbar.set_postfix({'epoch': f'{epoch+1}/{num_epochs}', 'Loss': f'{loss.item()}', 'RMSE': f'{torch.sqrt(loss):.4f}'})
    pred_y_train = linear_model(x_train_tensor)
    train_rmse = torch.sqrt(loss_fn(pred_y_train, y_train_tensor))
    print('train rmse:', train_rmse.item())

    pred_y_val = linear_model(x_val_tensor)
    val_rmse = torch.sqrt(loss_fn(pred_y_val, y_val_tensor))
    print('val rmse:', val_rmse.item())

    pred_y_test = linear_model(x_test_tensor)
    pred_y_test = pred_y_test.detach().numpy()
    return pred_y_test


def multi_layer_perceptron(x_train, y_train, x_val, y_val, x_test):
    x_train_tensor = torch.from_numpy(pd.get_dummies(x_train).values).float()
    y_train_tensor = torch.from_numpy(y_train.values).float().unsqueeze(1)
    x_val_tensor = torch.from_numpy(pd.get_dummies(x_val).values).float()
    y_val_tensor = torch.from_numpy(y_val.values).float().unsqueeze(1)
    x_test_tensor = torch.from_numpy(pd.get_dummies(x_test).values).float()
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    batch_size = len(x_train_tensor)
    train_dl = DataLoader(train_data, batch_size, shuffle=False)

    mlp_model = net = torch.nn.Sequential(
        torch.nn.Linear(18, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.01)
    num_epochs = 500

    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            for step, (xb, yb) in enumerate(train_dl):
                pred = mlp_model(xb)
                loss = loss_fn(pred, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pbar.update(1)
            pbar.set_postfix(
                {'epoch': f'{epoch + 1}/{num_epochs}', 'Loss': f'{loss.item()}', 'RMSE': f'{torch.sqrt(loss):.4f}'})
    pred_y_train = mlp_model(x_train_tensor)
    train_rmse = torch.sqrt(loss_fn(pred_y_train, y_train_tensor))
    print('train rmse:', train_rmse.item())

    pred_y_val = mlp_model(x_val_tensor)
    val_rmse = torch.sqrt(loss_fn(pred_y_val, y_val_tensor))
    print('val rmse:', val_rmse.item())

    pred_y_test = mlp_model(x_test_tensor)
    pred_y_test = pred_y_test.detach().numpy()
    return pred_y_test


def sklearn_linear_regression(x_train, y_train, x_val, y_val, x_test):
    linear_regression_estimator = LinearRegression()
    linear_regression_estimator.fit(x_train, y_train)

    pred_y_train = linear_regression_estimator.predict(x_train)
    train_rmse = math.sqrt(mean_squared_error(y_train, pred_y_train))
    print(f'train rmse: {train_rmse}')

    pred_y_val = linear_regression_estimator.predict(x_val)
    val_rmse = math.sqrt(mean_squared_error(y_val, pred_y_val))
    print(f'validation rmse: {val_rmse}')

    pred_y_test = linear_regression_estimator.predict(x_test)
    return pred_y_test


def linear_regression_with_lasso(x_train, y_train, x_val, y_val, x_test):
    lasso_estimator = Lasso(alpha=1.0, max_iter=1000, tol=0.0001)
    lasso_estimator.fit(x_train, y_train)

    pred_y_train = lasso_estimator.predict(x_train)
    train_rmse = math.sqrt(mean_squared_error(y_train, pred_y_train))
    print(f'train rmse: {train_rmse}')

    pred_y_val = lasso_estimator.predict(x_val)
    val_rmse = math.sqrt(mean_squared_error(y_val, pred_y_val))
    print(f'validation rmse: {val_rmse}')

    pred_y_test = lasso_estimator.predict(x_test)
    return pred_y_test


def linear_regression_with_ridge(x_train, y_train, x_val, y_val, x_test):
    ridge_estimator = Ridge(alpha=1.0)
    ridge_estimator.fit(x_train, y_train)

    pred_y_train = ridge_estimator.predict(x_train)
    train_rmse = math.sqrt(mean_squared_error(y_train, pred_y_train))
    print(f'train rmse: {train_rmse}')

    pred_y_val = ridge_estimator.predict(x_val)
    val_rmse = math.sqrt(mean_squared_error(y_val, pred_y_val))
    print(f'validation rmse: {val_rmse}')

    pred_y_test = ridge_estimator.predict(x_test)
    return pred_y_test


def decision_tree(x_train, y_train, x_val, y_val, x_test):
    decision_tree_regr = DecisionTreeRegressor(max_depth=25)
    decision_tree_regr.fit(x_train, y_train)

    pred_y_train = decision_tree_regr.predict(x_train)
    train_rmse = math.sqrt(mean_squared_error(y_train, pred_y_train))
    print(f'train rmse: {train_rmse}')

    pred_y_val = decision_tree_regr.predict(x_val)
    val_rmse = math.sqrt(mean_squared_error(y_val, pred_y_val))
    print(f'validation rmse: {val_rmse}')

    pred_y_test = decision_tree_regr.predict(x_test)
    return pred_y_test


def gradient_boosting_model(x_train, y_train, x_val, y_val, x_test):
    params = {
        "n_estimators": 500,
        "max_depth": 10,
        "min_samples_split": 5,
        "learning_rate": 0.01
    }
    gbm_reg = ensemble.GradientBoostingRegressor(**params)
    print('wait...(It might take a long time)')
    gbm_reg.fit(x_train, y_train)

    train_rmse = math.sqrt(mean_squared_error(y_train, gbm_reg.predict(x_train)))
    print(f'train rmse: {train_rmse}')

    val_rmse = math.sqrt(mean_squared_error(y_val, gbm_reg.predict(x_val)))
    print(f'validation rmse: {val_rmse}')

    pred_y_test = gbm_reg.predict(x_test)
    return pred_y_test


def xgboost_model(x_train, y_train, x_val, y_val, x_test):
    xgb_model = xgboost.XGBRegressor(n_estimators=2000, learning_rate=0.02, gamma=0, subsample=0.75,
                                     colsample_bytree=1, max_depth=6)
    print('wait...(It might take a long time)')
    xgb_model.fit(x_train, y_train)

    y_pred_train = xgb_model.predict(x_train)
    train_rmse = math.sqrt(mean_squared_error(y_train, y_pred_train))
    print(f'train rmse: {train_rmse}')

    y_pred_val = xgb_model.predict(x_val)
    val_rmse = math.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f'validation rmse: {val_rmse}')

    y_test_val = xgb_model.predict(x_test)
    return y_test_val


def light_gradient_boosting_model(x_train, y_train, x_val, y_val, x_test):
    train_ds = lgb.Dataset(x_train, label=y_train)
    test_ds = lgb.Dataset(x_val, label=y_val)

    params = {'learning_rate': 0.00009,
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
    lgb_model = lgb.train(params, train_ds, 1000, test_ds, verbose_eval=100, early_stopping_rounds=100)

    y_pred_train = lgb_model.predict(x_train)
    train_rmse = math.sqrt(mean_squared_error(y_train, y_pred_train))
    print(f'train rmse: {train_rmse}')

    y_pred_val = lgb_model.predict(x_val)
    val_rmse = math.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f'validation rmse: {val_rmse}')

    pred_y_test = lgb_model.predict(x_test)
    return pred_y_test


def create_kaggle_submit_csv(y_pred, y_test, file_name='submit_file.csv'):
    y_test['price'] = y_pred
    y_test = y_test[['id', 'date', 'price']]

    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'price'])
        for y_id, y_date, y_price in y_test.values:
            if len(y_id) != 10:
                y_id = '0' * (10 - len(y_id)) + y_id
            writer.writerow([y_id + y_date, y_price])
    print('file saved')


def main(**kwargs):
    args = Namespace(**kwargs)

    train_data, val_data, test_data = load_data()
    x_train = train_data
    x_train = x_train.drop(['id', 'date', 'price'], axis=1)
    y_train = train_data['price']
    x_val = val_data
    x_val = x_val.drop(['id', 'date', 'price'], axis=1)
    y_val = val_data['price']
    y_test = test_data
    x_test = test_data
    x_test = x_test.drop(['id', 'date', 'price'], axis=1)
    y_test = y_test.astype({'id':str})

    # linear model only 10 epoch
    if args.model == 'linear_10':
        y_pred = linear_model(x_train, y_train, x_val, y_val, x_test, 10)
    # linear model 500 epoch
    elif args.model == 'linear_500':
        y_pred = linear_model(x_train, y_train, x_val, y_val, x_test, 500)
    # multi layer perceptron
    elif args.model == 'mlp':
        y_pred = multi_layer_perceptron(x_train, y_train, x_val, y_val, x_test)
    # scikit-learn linear regression
    elif args.model == 'sklearn_linear_regression':
        y_pred = sklearn_linear_regression(x_train, y_train, x_val, y_val, x_test)
    # scikit-learn linear regression with lasso regularizer
    elif args.model == 'lasso':
        y_pred = linear_regression_with_lasso(x_train, y_train, x_val, y_val, x_test)
    # scikit-learn linear regression with ridge regularizer
    elif args.model == 'ridge':
        y_pred = linear_regression_with_ridge(x_train, y_train, x_val, y_val, x_test)
    # decision tree
    elif args.model == 'decision_tree':
        y_pred = decision_tree(x_train, y_train, x_val, y_val, x_test)
    # gradient boosting model
    elif args.model == 'gbm':
        y_pred = gradient_boosting_model(x_train, y_train, x_val, y_val, x_test)
    # xgboost
    elif args.model == 'xgboost':
        y_pred = xgboost_model(x_train, y_train, x_val, y_val, x_test)
    # light gradient boosting model
    elif args.model == 'lgbm':
        y_pred = light_gradient_boosting_model(x_train, y_train, x_val, y_val, x_test)
    else:
        print('Input error. please enter one of these ( linear_10 / linear_500 / mlp / sklearn_linear_regression / lasso / ridge / decision_tree / gbm / xgboost / lgbm )')
        sys.exit()

    create_kaggle_submit_csv(y_pred, y_test)

# you can run this file => python3 main.py linear_10
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model')
    args = p.parse_args()
    main(**vars(args))

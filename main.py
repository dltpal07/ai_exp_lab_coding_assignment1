import pandas as pd
import torch
import csv
from argparse import Namespace
import argparse
import sys
import torch_linear, multi_layer_perceptron, sklearn_linear_regression, linear_regression_with_lasso, linear_regression_with_ridge, \
    decision_tree, gradient_boosting_model, xgboost_model, light_gradient_boosting_model


def load_data(data_path, is_test=False):
    data = pd.read_csv(data_path)
    if is_test:
        y_data = data
    else:
        y_data = data['price']
    x_data = data.drop(['id', 'date', 'price'], axis=1)
    return x_data, y_data


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

    train_data_file_name = 'data/price_data_tr.csv'
    val_data_file_name = 'data/price_data_val.csv'
    test_data_file_name = 'data/price_data_ts.csv'

    x_train, y_train = load_data(train_data_file_name)
    x_val, y_val = load_data(val_data_file_name)
    x_test, y_test = load_data(test_data_file_name, is_test=True)
    y_test = y_test.astype({'id':str})

    if args.model == 'linear':
        model = torch_linear.Model()
    elif args.model == 'mlp':
        model = multi_layer_perceptron.Model()
    elif args.model == 'sklearn_linear_regression':
        model = sklearn_linear_regression.Model()
    elif args.model == 'lasso':
        model = linear_regression_with_lasso.Model()
    elif args.model == 'ridge':
        model = linear_regression_with_ridge.Model()
    elif args.model == 'decision_tree':
        model = decision_tree.Model()
    elif args.model == 'gbm':
        model = gradient_boosting_model.Model()
    elif args.model == 'xgboost':
        model = xgboost_model.Model()
    elif args.model == 'lgbm':
        model = light_gradient_boosting_model.Model()
    else:
        print('Input error. please enter one of these ( linear / mlp / sklearn_linear_regression / lasso / ridge / decision_tree / gbm / xgboost / lgbm )')
        sys.exit()
    model.train(x_train, y_train)
    y_train_pred = model.predict(x_train)
    train_rmse = model.rmse(y_train_pred, y_train)
    y_val_pred = model.predict(x_val)
    val_rmse = model.rmse(y_val_pred, y_val)
    print(f'train rmse: {train_rmse}\nval rmse: {val_rmse}')
    y_pred = model.predict(x_test)
    if isinstance(y_pred, type(torch.tensor(1))):
        y_pred = y_pred.detach().numpy()

    create_kaggle_submit_csv(y_pred, y_test)

# you can run this file => python3 main.py linear
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model')
    args = p.parse_args()
    main(**vars(args))

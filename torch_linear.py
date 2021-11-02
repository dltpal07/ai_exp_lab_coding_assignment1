import torch
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


class Model():
    def __init__(self):
        self.linear_model = torch.nn.Linear(18, 1)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.linear_model.parameters(), lr=0.01)
        self.num_epochs = 500
    def x_tensor_data(self, data):
        x_tensor = torch.from_numpy(pd.get_dummies(data).values).float()
        return x_tensor

    def y_tensor_data(self, data):
        y_tensor = torch.from_numpy(data.values).float().unsqueeze(1)
        return y_tensor

    def train(self, x_train, y_train):
        x_train = self.x_tensor_data(x_train)
        y_train = self.y_tensor_data(y_train)
        train_data = TensorDataset(x_train, y_train)
        batch_size = len(train_data)
        train_dl = DataLoader(train_data, batch_size, shuffle=False)

        with tqdm(total=self.num_epochs) as pbar:
            for epoch in range(self.num_epochs):
                for step, (xb, yb) in enumerate(train_dl):
                    pred = self.linear_model(xb)
                    loss = self.loss_fn(pred, yb)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                pbar.update(1)
                pbar.set_description(f'epoch: {epoch+1}/{self.num_epochs} rmse: {torch.sqrt(loss):.4f}')

    def predict(self, x_val):
        x_val = self.x_tensor_data(x_val)
        pred_y = self.linear_model(x_val)
        return pred_y

    def rmse(self, pred_y, true_y):
        true_y = self.y_tensor_data(true_y)
        return torch.sqrt(self.loss_fn(pred_y, true_y))
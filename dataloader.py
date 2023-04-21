import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# dataloader
class MyDataset(Dataset):
    def __init__(self, dataframe, device):
        self.data = dataframe.values
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index, :-10])
        y = self.data[index, -10:]
        return x.view(-1,28,28).float().to(self.device), torch.tensor(y, dtype=torch.float32).to(self.device)


def dataloader(device, dataloader_dic):

    train = pd.read_csv(dataloader_dic["train_name"])
    y_train = pd.get_dummies(train["label"],prefix="label")
    pred = pd.read_csv(dataloader_dic["test_name"])

    test = train.sample(frac = float(dataloader_dic["test_frac"]), random_state = 1)
    y_test = y_train.iloc[test.index]
    train.drop(test.index, inplace = True)
    y_train.drop(test.index, inplace = True)
    train.drop(columns =  ["label"], inplace = True)
    test.drop(columns =  ["label"], inplace = True)

    # Normalize
    train = train/ 255
    test = test / 255
    pred = pred / 255
    train = pd.concat([train,y_train],axis=1)
    test = pd.concat([test,y_test],axis=1)
    train_dataset = MyDataset(train, device)
    test_dataset = MyDataset(test, device)

    train_data_loader = DataLoader(train_dataset, batch_size=int(dataloader_dic["batch_size"]), shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=test.shape[0], shuffle=True)
    pred = torch.tensor(pred.values, dtype=torch.float32).view(-1,1,28,28).float().to(device)

    return train_data_loader,test_data_loader, pred


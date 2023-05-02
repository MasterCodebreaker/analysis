import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from my_models import Hook



def angle_loss(y_true, y_pred):
    

    # Calculate the angular distance (the minimum difference between angles)
    angle_diff = torch.abs(y_true - y_pred)
    angle_distance = torch.min(angle_diff, 1.0 - angle_diff)

    # Calculate the mean squared error loss
    mse_loss = nn.MSELoss()
    loss = mse_loss(angle_distance, torch.zeros_like(angle_distance))

    return loss


class NN():
    def __init__(self,model,training_dic, device, model_n, input_dim, starting_epoch: int = 1):
        # TODO REMOVE DEVICE
        self.train_acc = []
        self.test_acc = []
        self.train_loss = []
        self.layer_names  = []
        self.epok = starting_epoch
        self.model_n = model_n
        self.device = device
        self.input_dim = input_dim
        print("input_dim = ", input_dim)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=float(training_dic["lr"]), momentum=float(training_dic["momentum"]), weight_decay=float(training_dic["weight_decay"]))
        #self.criterion = nn.CrossEntropyLoss()
        #self.optimizer  = torch.optim.Adam(self.model.parameters())
        self.criterion = angle_loss
        self.scheduler = StepLR(self.optimizer, step_size=int(training_dic["step_size"]), gamma=float(training_dic["gamma"]))

        for name, _ in self.model.named_children():
            if not name.startswith("params"):
                self.layer_names.append(name)

    def train(self, n_epochs, train_data_loader, test_data_loader):
        for epoch in range(n_epochs):
            train_loss = 0
            number = 0
            train_acc = 0

            for data in tqdm(train_data_loader, desc = f"Epoch {self.epok + 1}") :
                x,y = data
                number += x.shape[0]
                output = self.model.forward(x)
                # scheduer step
                loss = self.criterion(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()/len(train_data_loader.dataset)*len(x)
                train_acc += sum(output.detach().cpu().argmax(axis = 1) == y.detach().cpu().argmax(axis = 1))
            # scheduer step
            self.scheduler.step()
            self.epok += 1
            
            with torch.no_grad():
                if len(test_data_loader) >0:
                    x,y = next(iter(test_data_loader))
                    test_x = self.model.forward(x)
                    test_acc = sum(test_x.detach().cpu().argmax(axis = 1) == y.detach().cpu().argmax(axis = 1))/x.detach().cpu().shape[0]
                else:
                    test_acc = 1
            
            self.train_loss.append(float(train_loss))
            self.train_acc.append(float(train_acc/number))
            self.test_acc.append(float(test_acc))
            print(f"Loss = {train_loss} , train_acc = {float(train_acc/number)} , test acc = {float(test_acc)}, lr = {self.optimizer.param_groups[0]['lr']}, epoch = {self.epok}")
            
        with torch.no_grad():
            hookF = [Hook(layer[1]) for layer in list(self.model._modules.items())]
            # TODO: Not the most elegant way of doing it, avoid pushing dataset through a second time, or do I have to?
            self.model.forward(torch.tensor(train_data_loader.dataset.data[:, :-1]).view(-1,self.input_dim[1],self.input_dim[2]).float().to(self.device))
            # Save activations with the hooks traindata.
            for i,hook in enumerate(hookF):
                name = self.layer_names[i]
                np.save("./activations/"+ self.model_n + "_"+ name +"_"+str(self.epok)+ "_"+ ".npy", hook.output.detach().cpu().numpy())

            # Saving weights from network
            for name in self.layer_names:
                try:
                    w = getattr(self.model, name).weight.detach().cpu().numpy()
                    np.save("./weights/"+self.model_n+"_"+ name +"_"+str(self.epok)+ "_"+ ".npy",w)
                except:
                    # This implies there is no weights in the layer, such as ReLu.
                    pass 


    def training(self, training_bulks, train_data_loader, test_data_loader):
        for n_epochs in training_bulks:
            #self.epok += n_epochs
            self.train(n_epochs = n_epochs,train_data_loader = train_data_loader, test_data_loader = test_data_loader)
            torch.save(self.model.state_dict(), "./models/" +self.model_n+ "_"+ str(self.epok)+".pt")
            print("saving model as: "+ "./models/"+self.model_n+"_"+ str(self.epok)+".pt")

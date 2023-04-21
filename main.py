from my_models import *
from configparser import ConfigParser
from dataloader import dataloader
from torchsummary import summary
from network import NN
from utils import parse_int_tuple
config_file = "./settings1.ini"

parser = ConfigParser()
parser.read(config_file)

#DataLoader
dataloader_dic = parser["dataloader"]

# Network
network_dic = parser["network"]

#training
training_dic = parser["training"]


if __name__ == "__main__":
    model_name = "models/"+ network_dic["model_n"]+"_"+network_dic["starting_epoch"]+".pt"
    #If server
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #If local
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    #DataLoader 
    train_data_loader, test_data_loader, pred = dataloader(device=device, dataloader_dic=dataloader_dic)
    
    #Create and train
    if network_dic["model_n"] == "Net0":
        model = Net0()
    else:
        print("There exist no model with this name")
    #remove
    print(summary(model,parse_int_tuple(network_dic["input_dimension"])))
    print(f"Device = {device}, training_bulks = ", training_dic["training_bulks"])
    try: 
        model.load_state_dict(torch.load(model_name))
        print("Loaded old model: "+ model_name)
    except:
        print("Cant load old model, training from scratch")
    model.to(device)

    network = NN(model = model,device = device,training_dic = training_dic, model_n = network_dic["model_n"] , starting_epoch=int(network_dic["starting_epoch"]))
    network.training(training_bulks=parse_int_tuple(training_dic["training_bulks"]), train_data_loader=train_data_loader, test_data_loader=test_data_loader)

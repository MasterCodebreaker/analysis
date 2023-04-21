from configparser import ConfigParser
"""
Make your own config file.
"""
config = ConfigParser()

# DataLoader variables
config["dataloader"] = {
    "train_name": "./data/train.csv",
    "test_name": "./data/test.csv",
    "test_frac": 0.1,
    "batch_size": 128,
}
# Network variables
config["network"] = {
    "input_dimension": (1,28,28),
    "model_n" : "Net0",
    "starting_epoch": 0,
}
# Training variables
config["training"] = {
    "lr" : 0.1,
    "weight_decay" : 0.0001,
    "momentum" : 0.9,
    "gamma" : 0.1,
    "step_size" : 100,
    "training_bulks" : (1,2)
}
with open("./settings1.ini", "w") as f:
    config.write(f)


from configparser import ConfigParser
"""
Make your own config file.
"""
config = ConfigParser()

# DataLoader variables
config["dataloader"] = {
    "train_name": "/Users/sigurgau/Documents/activations_mnist/rotated_lines.csv",
    "test_name": "nan",
    "test_frac": 0.1,
    "batch_size": 128,
}
# Network variables
config["network"] = {
    "input_dimension": (1,29,29),
    "model_n" : "Net1",
    "starting_epoch": 0,
}
# Training variables
config["training"] = {
    "lr" : 0.1,
    "weight_decay" : 0.0001,
    "momentum" : 0.9,
    "gamma" : 0.1,
    "step_size" : 100,
    "training_bulks" : (10,40,50,50,50,50,50,50)
}
with open("./settings_net1.ini", "w") as f:
    config.write(f)


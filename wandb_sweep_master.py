from train import train
import os
import json

def main():

    sweep_config_1 = {
                'method': 'grid',
                'metric': {'goal': 'minimize', 'name': 'val_loss'},
                'parameters': {
                    'batch_size': {'value': 32},
                    'epochs': {'values': [5, 10]},
                    'lr': {'values': [1e-2, 1e-3, 1e-4]},
                    'model': {'value': 'ResNet18'}
                    }
    }

    sweep_config_2 = {
                'method': 'grid',
                'metric': {'goal': 'minimize', 'name': 'val_loss'},
                'parameters': {
                    'batch_size': {'value': 64},
                    'epochs': {'values': [5, 10]},
                    'lr': {'values': [1e-2, 1e-3, 1e-4]},
                    'model': {'value': 'ResNet18'}
                    }
    }

    sweep_config_3 = {
                'method': 'grid',
                'metric': {'goal': 'minimize', 'name': 'val_loss'},
                'parameters': {
                    'batch_size': {'value': 128},
                    'epochs': {'values': [5, 10]},
                    'lr': {'values': [1e-2, 1e-3, 1e-4]},
                    'model': {'value': 'ResNet18'}
                    }
    }

    with open("configs/config1.json", "w") as outfile:
        json.dump(sweep_config_1, outfile)

    with open("configs/config2.json", "w") as outfile:
        json.dump(sweep_config_2, outfile)

    with open("configs/config3.json", "w") as outfile:
        json.dump(sweep_config_3, outfile)
    

    return

if __name__ == "__main__":
    main()
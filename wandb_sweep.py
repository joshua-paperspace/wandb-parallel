from train import train
import wandb
import os
import json
import sys, getopt

def main(args):

    try:
        opts, args = getopt.getopt(argv,"c:",["config="])
    except:
        print("Could not get any passed arguments.")
        print("Cannot continue without model.")
        return
    for opt, arg in opts:
        if opt in ('-c', '--config'):
            config_name = arg

    with open('/inputs/configs/' + config_name + '.json') as json_file:
        sweep_config = json.load(json_file)

    sweep_id = wandb.sweep(sweep_config, project="prallel-project")
    wandb.agent(sweep_id, function=train, count=4)

    return

if __name__ == "__main__":
    # main()
   main(sys.argv[1:])
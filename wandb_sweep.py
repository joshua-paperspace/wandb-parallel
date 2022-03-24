from train import train
import wandb
import os
import json

def main():

    # try:
    #     opts, args = getopt.getopt(argv,"s:",["sweepId="])
    # except:
    #     print("Could not get any passed arguments.")
    #     print("Cannot continue without model.")
    #     return
    # for opt, arg in opts:
    #     if opt in ('-s', '--sweepId'):
    #         sweep_id = arg

    with open('/inputs/config.json') as json_file:
        sweep_config = json.load(json_file)

    sweep_id = wandb.sweep(sweep_config, project="prallel-project")
    wandb.agent(sweep_id, function=train, count=4)

    return

if __name__ == "__main__":
    main()
#    main(sys.argv[1:])
import time
import torch.nn as nn
import torch.optim as optim
import torch

import wandb
import os

from resnet import resnet18, resnet34
from load_data import load_data

def validate_model(model, valid_dl, loss_func, device):
    
    # Compute performance of the model on the validation dataset
    model.eval()
    val_loss = 0.

    with torch.inference_mode():

        correct = 0
        for i, (images, labels) in enumerate(valid_dl, 0):
            
            # Move data to GPU if available 
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            val_loss += loss_func(outputs, labels)*labels.size(0)

            # Compute accuracy and accumulate
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)
    
def train(config = None):
    with wandb.init(project='test-project', config=config):
        config = wandb.config

        trainloader, valloader, testloader = load_data(config)

        if config['model']=='ResNet18':
            model = resnet18(3,10)
        else:
            model = resnet34(3,10)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        step = 0
        batch_checkpoint=50
        epoch_durations = []
        for epoch in range(config['epochs']):
            
            epoch_start_time = time.time()
            running_loss = 0.0
            model.train()

            for i, data in enumerate(trainloader, 0):
            
                # Move data to GPU if available 
                inputs, labels = data[0].to(device), data[1].to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward + Backward + Optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
                running_loss += loss.item()
                
                # Log every 50 batches
                if i % batch_checkpoint == batch_checkpoint-1:
                    step +=1
                    print(f'epoch: {epoch + ((i+1)/len(trainloader)):.2f}')
                    wandb.log({"train_loss": running_loss/batch_checkpoint, "epoch": epoch + ((i+1)/len(trainloader))}, step=step)
                
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / batch_checkpoint))
                        
                    running_loss = 0.0
            
            # Log at the end of each epoch
            step +=1
            print(f'epoch: {epoch + ((i+1)/len(trainloader)):.2f}')
            wandb.log({"train_loss": running_loss/batch_checkpoint, "epoch": epoch + ((i+1)/len(trainloader))}, step=step)
                
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / batch_checkpoint))

            # Log validation metrics
            val_loss, accuracy = validate_model(model, valloader, criterion, device)
            wandb.log({"val_loss": val_loss, "val_accuracy": accuracy}, step=step)
            print(f"Valid Loss: {val_loss:3f}, accuracy: {accuracy:.2f}")
            
            epoch_duration = time.time() - epoch_start_time
            wandb.log({"epoch_runtime (seconds)": epoch_duration}, step=step)

            epoch_durations.append(epoch_duration)

        avg_epoch_runtime = sum(epoch_durations) / len(epoch_durations)
        wandb.log({"avg epoch runtime (seconds)": avg_epoch_runtime})

        #Upload model artifact to Gradient and log model name to W&B
        full_model_name = upload_model(config, model_client)
        wandb.log({"Notes": full_model_name})

    print('Training Finished')
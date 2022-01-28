from __future__ import print_function
from typing import Dict
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import wandb

import yaml
from pathlib import Path

def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    return parser.parse_args()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args: argparse.Namespace, model: nn.Module, device: torch.device, train_loader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int):
    model.train()
    with wandb.init():
        for idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if idx % args.log_interval == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, idx*len(data), len(train_loader.dataset),
                    100.*idx/len(train_loader), loss.item(),
                ))
                wandb.log({"loss": loss.item()})

def test(model: nn.Module, device: torch.device, test_loader: DataLoader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100.*correct/len(test_loader.dataset),
    ))

def load_sweep_config(path: str) -> Dict:
    path = Path(path)

    with path.open("r") as f:
        d = yaml.safe_load(f)

    return d

def main():
    sweep_config = {
        "entity": "shunpoco",
        "project": "test",
        "name" : "my-sweep",
        "method" : "random",
        "parameters" : {
            "epochs" : {
            "values" : [10, 20, 50]
            },
            "learning_rate" :{
            "min": 0.0001,
            "max": 0.1
            }
        }
    }
    # Neither using dict nor load from yaml file, we can't inherit entity and project to sweep.
    # yaml_path = "./sweep.yaml"
    # sweep_config = load_sweep_config(yaml_path)

    arg = args()
    sweep_id = wandb.sweep(sweep_config)

    use_cuda = not arg.no_cuda and torch.cuda.is_available()
    torch.manual_seed(arg.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": arg.batch_size}
    test_kwargs = {"batch_size": arg.test_batch_size}

    if use_cuda:
        cuda_kwargs = {
            "num_workers": 1,
            "pin_memory": True,
            "shuffle": True,
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_data = datasets.MNIST("../data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, **train_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=arg.lr)

    def t():
        return  train(arg, model, device, train_loader, optimizer, 1)

    wandb.agent(sweep_id, function=t, count=1)
    if arg.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == "__main__":
    main()

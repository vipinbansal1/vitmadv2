import argparse, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
import os
from PIL import Image


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        self.resnet = torchvision.models.resnet18(weights=None)

        self.fc_in_features = self.resnet.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)
        
        return output

class APP_MATCHER(Dataset):
    def __init__(self, root, train, download=False):
        super(APP_MATCHER, self).__init__()
        self.path_orig = root + 'classification/orig/'
        self.path_pred = root + 'classification/pred/'
               
        self.listfiles = os.listdir(self.path_orig)
        self.total_content = len(self.listfiles)
        self.series = list(range(0,536))
        
        random.shuffle(self.series)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        
        img_index = self.series[index]
        
        filename = self.listfiles[img_index]
        orig_filename = self.path_orig+filename
        pred_filename = self.path_pred+filename
        orig_img = np.array(Image.open(orig_filename))
        orig_img = np.einsum('hwc->chw',orig_img)
        mean = orig_img.mean((1,2))
        std = orig_img.std((1,2))
        if mean.ndim == 1:
            mean = mean.reshape(3,1,1)
        if std.ndim == 1:
            std = std.reshape(3, 1, 1)
        orig_img = (orig_img-mean)/std        
        orig_img = orig_img.astype('float32')
        orig_img = torch.from_numpy(orig_img)
        
        
        pred_img = np.array(Image.open(pred_filename))
        pred_img = np.einsum('hwc->chw',pred_img)
        mean = pred_img.mean((1,2))
        std = pred_img.std((1,2))
        if mean.ndim == 1:
            mean = mean.reshape(3,1,1)
        if std.ndim == 1:
            std = std.reshape(3, 1, 1)
        pred_img = (pred_img-mean)/std
        pred_img = pred_img.astype('float32')
        pred_img = torch.from_numpy(pred_img)
        
        label = -1        
        temp = filename.split("_")[0]
        if temp == 'A':
            label = 0
        else:
            label = 1
        target = torch.tensor(label, dtype=torch.float)
        
        return orig_img, pred_img, target
        


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    criterion = nn.BCELoss()

    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)
    #use_cuda = True
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_dataset = APP_MATCHER('./data/', train=True, download=True)
    test_dataset = APP_MATCHER('./data/', train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = SiameseNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        random.shuffle(train_dataset.series)
        test(model, device, test_loader)
        scheduler.step()

        if epoch%20 == 0:
           name = "siamese_network_cpu_with_diff_chisquare_"+str(epoch)+".pt"
           torch.save(model.state_dict(), name)


if __name__ == '__main__':
    main()

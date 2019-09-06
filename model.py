import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from configuration import Config
from data import ImageGenerator
from PIL import Image
import cv2
import numpy as np
from lr_finder import LRFinder
# from tqdm import tqdm
import time
import matplotlib.pyplot as plt
# import copy
from skimage import exposure
import warnings
import os
import random
import multiprocessing
multiprocessing.set_start_method('spawn', True)

cfg = Config()


class TrafficSignNet(nn.Module):
    def __init__(self):
        super(TrafficSignNet, self).__init__()
        self.stn = Stn()
        self.conv1 = nn.Conv2d(3, 100, 5)
        self.conv1_bn = nn.BatchNorm2d(100)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(100, 150, 3)
        self.conv2_bn = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, 1)
        self.conv3_bn = nn.BatchNorm2d(250)
        self.fc1 = nn.Linear(250 * 3 * 3, 350)
        self.fc1_bn = nn.BatchNorm1d(350)
        self.fc2 = nn.Linear(350, 43)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.stn(x)
        x = self.pool(F.elu(self.conv1(x)))
        x = self.dropout(self.conv1_bn(x))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.dropout(self.conv2_bn(x))
        x = self.pool(F.elu(self.conv3(x)))
        x = self.dropout(self.conv3_bn(x))
        x = x.view(-1, 250 * 3 * 3)
        x = F.elu(self.fc1(x))
        x = self.dropout(self.fc1_bn(x))
        x = self.fc2(x)
        return x


class Stn(nn.Module):
    def __init__(self):
        super(Stn, self).__init__()
        # Spatial transformer localization-network
        self.loc_net = nn.Sequential(
            nn.Conv2d(3, 50, 7),
            nn.MaxPool2d(2, 2),
            nn.ELU(),
            nn.Conv2d(50, 100, 5),
            nn.MaxPool2d(2, 2),
            nn.ELU()
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(100 * 4 * 4, 100),
            nn.ELU(),
            nn.Linear(100, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.loc_net(x)
        xs = xs.view(-1, 100 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x


class Resnet_model(nn.Module):

    def __init__(self):
        super(Resnet_model, self).__init__()

        self.resnet = torchvision.models.resnet34(pretrained=True)
        self.out = nn.Linear(1000, cfg.n_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.out(x)

        return F.softmax(x, dim=-1)


class TrainModel:

    def __init__(self, model):

        self.model = model
        self.model.to(cfg.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        # def lambda1(epoch):
            
        #     if epoch == 1:
        #         lr = self.optimizer.param_groups[0]['initial_lr']
        #     else:
        #         lr = self.optimizer.param_groups[0]['lr'] * 0.9

        #     return lr
        
        # self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        self.scheduler_initial = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 4000, eta_min=0, last_epoch=-1)
        self.writer = SummaryWriter('logs/{}'.format(time.time()))
        # visualize graph
        # data = torch.zeros((cfg.batch_size, cfg.image_c,
        #                     cfg.image_h, cfg.image_w)).to(cfg.device)
        # self.writer.add_graph(self.model, data)
        self.train_gen = ImageGenerator(input_path=cfg.train_dir, num_images=cfg.num_train_images,
                                        transform=cfg.data_transforms['train'])
        self.val_gen = ImageGenerator(input_path=cfg.test_dir, num_images=cfg.num_test_images,
                                      transform=cfg.data_transforms['val'])
        self.data_loaders = {
            'train': DataLoader(self.train_gen, batch_size=cfg.batch_size, num_workers=cfg.n_workers, shuffle=True),
            'val':  DataLoader(self.val_gen, batch_size=cfg.batch_size, num_workers=cfg.n_workers, shuffle=True)
        }

        if not os.path.exists(os.path.join(os.getcwd(), 'weights')):
            os.makedirs(os.path.join(os.getcwd(), 'weights'))

    def lr_finder(self, end_lr):

        lr_find = LRFinder(self.model, self.optimizer, self.criterion, cfg.device)
        lr_find.range_test(self.data_loaders['val'], end_lr=end_lr, num_iter=2000)
        lr_find.plot()

    def train(self, epoch):

        since = time.time()
        self.model.train()
        loss_previous = 100.
        for batch_idx, sample_batched in enumerate(self.data_loaders['train']):
            data, target = sample_batched['image'].to(cfg.device), sample_batched['label'].type(torch.LongTensor).to(
                cfg.device)  # LongTensor
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            if epoch == 1:
                self.scheduler_initial.step()
            
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} LR: {}'.format(
                    epoch, batch_idx, len(self.data_loaders['train']),
                    100. * batch_idx / len(self.data_loaders['train']), loss.item(), self.optimizer.param_groups[0]['lr']))
                self.writer.add_scalar('train loss',
                                       loss.item(),
                                       epoch * cfg.num_train_images // cfg.batch_size + batch_idx)

        time_elapsed = time.time() - since
        print('Train Epoch: {} complete in {:.0f}m {:.0f}s'.format(epoch,
                                                                   time_elapsed // 60, time_elapsed % 60))

    def val(self, epoch):
        
        self.model.eval()
        test_loss = 0.
        correct = 0.
        with torch.no_grad():
            for sample_batched in self.data_loaders['val']:
                data, target = sample_batched['image'].to(cfg.device), sample_batched['label'].type(torch.LongTensor).to(
                    cfg.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()  # sum up batch loss
                pred = torch.squeeze(output.max(1, keepdim=True)[1])
                correct += torch.sum(pred == target.data)
        test_loss /= len(self.data_loaders['val'])
        val_acc = correct.item() / (len(self.data_loaders['val']) * cfg.batch_size)
        self.writer.add_scalar('val loss',
                               test_loss,
                               epoch)
        self.writer.add_scalar('val_accuracy',
                               val_acc,
                               epoch)
        print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)\n'.format(
            test_loss, int(correct / cfg.batch_size), len(self.data_loaders['val']), val_acc))
        torch.save(self.model.state_dict(), os.path.join(os.getcwd(), 'weights', 'val_acc_{}.pt'.format(val_acc)))

        return None

    def test(self, image_path):

        self.model.eval()
        with torch.no_grad():
            image = cv2.imread(image_path)
            raw_image = image.copy()
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (cfg.image_h, cfg.image_w), interpolation=cv2.INTER_AREA)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                image = exposure.equalize_adapthist(image)
            image = image.astype("float32")
            image /= 255
            image = (image - cfg.mean) / cfg.std
            image = np.transpose(image, (2, 0, 1))
            image = torch.unsqueeze(torch.tensor(image, dtype=torch.float, device=cfg.device), 0)
            output = self.model(image)
            pred = output.max(1, keepdim=True)[1]
            pred = torch.squeeze(pred).to(torch.device('cpu'))
            print(pred.numpy())
            plt.imshow(raw_image)
            plt.show()


def train_model(pretrained_weights=None):

    model = TrafficSignNet()
    # print(model)
    if pretrained_weights is not None:
        model.load_state_dict(torch.load(pretrained_weights))
    trainer = TrainModel(model)
    # trainer.lr_finder(1e-2) # function plot initial learning rate
    for epoch in range(1, cfg.n_epochs + 1):
        trainer.train(epoch)
        trainer.val(epoch)
        trainer.scheduler.step()

def test_model(image_path, pretrained_weights='weights/val_acc_0.9739622114668652.pt'):

    model = TrafficSignNet()
    if pretrained_weights is not None:
        model.load_state_dict(torch.load(pretrained_weights))
    trainer = TrainModel(model)
    trainer.test(image_path)


if __name__ == '__main__':
    image_path = 'testing_images/00000/00000_00000.ppm'
    # train_model()
    test_model(image_path)
    print('complete')

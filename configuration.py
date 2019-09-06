import torch
from torchvision import transforms
import os
import cv2
from PIL import Image
import numpy as np


class Config:

    def __init__(self):
        # image parameters
        self.image_h = 32
        self.image_w = 32
        self.image_c = 3
        # text files
        self.resized_img_num_train = 10000
        self.resized_img_num_test = 1000
        if os.path.exists("res_train.txt") and os.path.exists("res_test.txt"):
            self.train_dir = os.path.abspath("res_train.txt")
            self.test_dir = os.path.abspath("res_test.txt")
            with open(self.train_dir, "r") as f:
                self.num_train_images = len(f.readlines())
            with open(self.test_dir, "r") as f:
                self.num_test_images = len(f.readlines())
        # model parameters
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_epochs = 10
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.mean = [0.32389543, 0.31369685, 0.33691839]
        self.std  = [0.17405162, 0.1679329,  0.17242609]
        self.n_classes = 43
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor()
            ]),
            'val': transforms.Compose([
                transforms.ToTensor()
            ])
        }
        self.n_workers = 4
        # self.weight_coef = [
        #     2.7649167033667936, 0.5708059347902017, 0.5512156600707678, 1.4523682971647254, 0.8450697808622758,
        #     0.962611429178879, 2.5298334067335873, 1.4327780224452915, 1.4523682971647254, 1.3935974730064238,
        #     0.8058892314234081, 1.5503196707618945, 0.7079378578262387, 0.6491670336679372, 2.1380279123449104,
        #     2.314340384819815, 2.5298334067335873, 1.7854029673951008, 1.6874515937979315, 2.7649167033667936,
        #     2.6081945056113227, 2.6277847803307566, 2.569013956172455, 2.431882033136418, 2.706145879208492,
        #     1.3544169235675563, 2.333930659539249, 2.725736153927926, 2.412291758416984, 2.706145879208492,
        #     2.5102431320141534, 2.1380279123449104, 2.725736153927926, 2.236632295099394, 2.5298334067335873,
        #     1.6874515937979315, 2.569013956172455, 2.7649167033667936, 0.7471184072651065, 2.6669653297696243,
        #     2.6081945056113227, 2.725736153927926, 2.725736153927926
        # ]

    def computing_mean_std(self):

        _mean_absolut = np.array([0.0, 0.0, 0.0])
        _std_absolut = np.array([0.0, 0.0, 0.0])
        file_path = self.train_dir

        print('start: {}'.format(file_path))
        with open(file_path, 'r') as f:
            img_list = f.readlines()

        for line in img_list:
            line = line.split(';')
            x_data = cv2.imread(line[0])
            x_data = cv2.resize(x_data, (self.image_h, self.image_w))
            x_data = x_data.astype("float32")
            x_data /= 255
            means = x_data.mean(axis=(0, 1), dtype='float64')
            stds = x_data.std(axis=(0, 1), dtype='float64')

            _mean_absolut += means
            _std_absolut += stds
            print(line[0])

        _mean_absolut /= self.num_train_images
        _std_absolut /= self.num_train_images

        with open('mean_std.txt', 'w') as f:
            f.write('mean: {}\nstd: {}'.format(_mean_absolut, _std_absolut))
        return None


if __name__ == '__main__':
    cfg = Config()
    cfg.computing_mean_std()

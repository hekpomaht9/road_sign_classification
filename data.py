from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import glob
import json
from skimage import exposure
import warnings

from configuration import Config
import multiprocessing

multiprocessing.set_start_method('spawn', True)
cfg = Config()


class ImageGenerator(Dataset):

    def __init__(self, input_path, num_images, transform=None):

        with open(input_path, 'r') as f:
            self.img_list = f.readlines()
        self.num_images = num_images
        self.transform = transform

    def __len__(self):
        return (self.num_images // cfg.batch_size) * cfg.batch_size

    def __getitem__(self, idx):
        # read img_list with random index
        line = self.img_list[idx].strip().split(';')
        # read as image
        image = cv2.imread(line[0])
        # constrate
        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                image = exposure.equalize_adapthist(image)
        # normalisation
        image = image.astype("float32")
        image /= 255
        # standardization
        image = (image - cfg.mean) / cfg.std
        # transform to tensor
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float)
        label = int(line[-1])
        label = torch.tensor(np.array(label), dtype=torch.long)

        return {'image': image, 'label': label}


def counting_classes(input_path):

    with open(input_path, 'r') as f:
        img_list = f.readlines()

    counter_classes = np.zeros(cfg.n_classes)

    for line in img_list:
        class_idx = line.strip().split(';')[-1]
        counter_classes[int(class_idx)] += 1

    ind = np.arange(cfg.n_classes)
    p = plt.bar(ind, counter_classes)

    # plt.show()
    print(counter_classes)
    weight_coefs = 3 - (counter_classes / len(img_list) * cfg.n_classes)
    weight_coefs = weight_coefs.tolist()
    print(weight_coefs)

    with open('weight_coef.txt', 'w') as f:
        f.write(', '.join(map(str, weight_coefs)))
    return None


class BalancedData:

    def __init__(self, input_path, phase):

        with open(input_path, 'r') as f:
            self.img_list = f.readlines()
        self.num_imgs = len(self.img_list)
        self.label_counter = {}# label: number, img_name, rotation_angle, gamma
        for i in range(cfg.n_classes):
            self.label_counter[i] = {}
        self.end_flag = np.zeros(cfg.n_classes)
        self.phase = phase
        if self.phase == 'train':
            self.num_res_images = cfg.resized_img_num_train
        elif self.phase == 'test':
            self.num_res_images = cfg.resized_img_num_test

    def append_dict(self, img_name, img_label, img_angle, img_gamma):

        self.label_counter[img_label][len(self.label_counter[img_label])] = {
            'name': img_name,
            'angle': img_angle,
            'gamma': img_gamma
        }

    def create_config_dict(self):

        while int(sum(self.end_flag)) < 43:
            for line in self.img_list:

                line = line.strip()

                img_name = line.split(';')[0]
                img_label = int(line.split(';')[-1])
                img_angle = random.randint(-20, 20)
                img_gamma = random.uniform(0.5, 1.5)

                print(img_name + '  ' + str(sum(self.end_flag)))

                if len(self.label_counter[img_label]) == self.num_res_images:
                    self.end_flag[img_label] = 1
                    continue
                else:
                    for data in self.label_counter[img_label].keys():
                        data = self.label_counter[img_label][data]
                        if data['name'] == img_name  and\
                           data['angle'] == img_angle and data['gamma'] == img_gamma:
                            continue
                    self.append_dict(img_name, img_label, img_angle, img_gamma)

        with open('resized_data_{}.json'.format(self.phase), 'w') as fp:
            json.dump(self.label_counter, fp)
        print('complete')

    def create_resized_images(self):

        input_path='resized_data_{}.json'.format(self.phase)
        assert os.path.exists(input_path)

        with open(input_path, 'r') as f:
            data_json = json.load(f)

        for image_label in data_json.keys():
            if not os.path.exists(os.path.join('resize_{}'.format(self.phase), image_label)):
                os.makedirs(os.path.join('resize_{}'.format(self.phase), image_label))
            for image_idx in data_json[image_label].keys():
                print(image_idx)
                out_path = os.path.join('resize_{}'.format(self.phase), image_label, '{0:05}.jpg'.format(int(image_idx)))
                # read as image
                image = cv2.imread(data_json[image_label][image_idx]['name'])
                image = cv2.resize(image, (cfg.image_h, cfg.image_w), interpolation=cv2.INTER_AREA)
                # brightness
                look_ip_table = np.empty((1, 256), np.uint8)
                for i in range(256):
                    look_ip_table[0, i] = np.clip(pow(i / 255.0,
                                                    data_json[image_label][image_idx]['gamma']) * 255.0, 0, 255)
                image = cv2.LUT(image, look_ip_table)
                # rotation
                m = cv2.getRotationMatrix2D((cfg.image_h / 2, cfg.image_w / 2),
                                            data_json[image_label][image_idx]['angle'], 1.)
                image = cv2.warpAffine(image, m, (cfg.image_h, cfg.image_w))
                # save
                cv2.imwrite(out_path, image)

                with open('res_{}.txt'.format(self.phase), 'a') as f:
                    f.write(out_path + ';{}\n'.format(image_label))


class FileCreator:

    def __init__(self):

        self.dirs = {
            'train': ['D:/task6/train_dataset/Training_1',
                      'D:/task6/train_dataset/Training_2'],

            'test':  ['D:/task6/test_dataset/Testing_1']
        }
        self.file = 0

    def create_txt(self, phase):

        for dir in self.dirs[phase]:

            folders = os.listdir(dir)

            for folder in folders:

                self.file = glob.glob(os.path.join(dir, folder, '*.csv'))
                self.file = self.file[0]

                with open(self.file) as csvfile:
                    lines = csvfile.readlines()

                for i, line in enumerate(lines, 0):
                    if i > 0:
                        with open('{}.txt'.format(phase), 'a') as f:
                            f.write(os.path.join(dir, folder, line))
                print(folder)

        return None


if __name__ == '__main__':

    creator = FileCreator()
    for phase in ['train', 'test']:
        creator.create_txt(phase)
        
        bd = BalancedData('{}.txt'.format(phase), phase)
        bd.create_config_dict()
        bd.create_resized_images()

    cfg.train_dir = os.path.abspath("res_train.txt")
    cfg.test_dir = os.path.abspath("res_test.txt")
    with open(cfg.train_dir, "r") as f:
        cfg.num_train_images = len(f.readlines())
    with open(cfg.test_dir, "r") as f:
        cfg.num_test_images = len(f.readlines())
    cfg.computing_mean_std()
    counting_classes(cfg.train_dir)

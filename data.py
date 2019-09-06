from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os
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

    with open('weight_coef', 'w') as f:
        f.write(', '.join(map(str, weight_coefs)))
    return None


class BalancedData:

    def __init__(self, input_path=cfg.train_dir):

        with open(input_path, 'r') as f:
            self.img_list = f.readlines()
        self.num_imgs = len(self.img_list)
        self.label_counter = {}# label: number, img_name, flip, rotation_angle, gamma
        for i in range(cfg.n_classes):
            self.label_counter[i] = {}
        self.end_flag = np.zeros(cfg.n_classes)

    def append_dict(self, img_name, img_label, img_flip, img_angle, img_gamma):

        self.label_counter[img_label][len(self.label_counter[img_label])] = {
            'name': img_name,
            'flip': img_flip,
            'angle': img_angle,
            'gamma': img_gamma
        }

    def create_config_dict(self):

        while int(sum(self.end_flag)) < 43:
            for line in self.img_list:

                line = line.strip()

                img_name = line.split(';')[0]
                img_label = int(line.split(';')[-1])
                img_flip = True if random.random() > 0.5 else False
                img_angle = random.randint(-20, 20)
                img_gamma = random.uniform(0.5, 1.5)

                print(img_name + '  ' + str(sum(self.end_flag)))

                if img_label not in self.label_counter.keys():
                    self.append_dict(img_name, img_label, img_flip, img_angle, img_gamma)
                elif len(self.label_counter[img_label]) == cfg.resized_img_num:
                    self.end_flag[img_label] = 1
                    continue
                else:
                    for data in self.label_counter[img_label].keys():
                        data = self.label_counter[img_label][data]
                        if data['name'] == img_name and data['flip'] == img_flip and\
                           data['angle'] == img_angle and data['gamma'] == img_gamma:
                            continue
                    self.append_dict(img_name, img_label, img_flip, img_angle, img_gamma)

        with open('resized_data.json', 'w') as fp:
            json.dump(self.label_counter, fp)
        print('complete')

    def create_resized_images(self, input_path='resized_data.json'):

        assert os.path.exists(input_path)

        with open(input_path, 'r') as f:
            data_json = json.load(f)

        for image_label in data_json.keys():
            if not os.path.exists(os.path.join('resize', image_label)):
                os.makedirs(os.path.join('resize', image_label))
            for image_idx in data_json[image_label].keys():
                print(image_idx)
                out_path = os.path.join('resize', image_label, '{0:05}.jpg'.format(int(image_idx)))
                # read as image
                image = cv2.imread(data_json[image_label][image_idx]['name'])
                image = cv2.resize(image, (cfg.image_h, cfg.image_w), interpolation=cv2.INTER_AREA)
                # brightness
                look_ip_table = np.empty((1, 256), np.uint8)
                for i in range(256):
                    look_ip_table[0, i] = np.clip(pow(i / 255.0,
                                                    data_json[image_label][image_idx]['gamma']) * 255.0, 0, 255)
                image = cv2.LUT(image, look_ip_table)
                # flip
                if data_json[image_label][image_idx]['flip']:
                    image = cv2.flip(image, 0)
                # rotation
                m = cv2.getRotationMatrix2D((cfg.image_h / 2, cfg.image_w / 2),
                                            data_json[image_label][image_idx]['angle'], 1.)
                image = cv2.warpAffine(image, m, (cfg.image_h, cfg.image_w))
                # save
                cv2.imwrite(out_path, image)

                with open('resized_images.txt', 'a') as f:
                    f.write(out_path + ';{}\n'.format(image_label))

    def create_text_files(self, input_path='resized_images.txt'):

        assert os.path.exists(input_path)

        with open(input_path, 'r') as f:
            image_list = f.readlines()

        image_list.sort()
        counter = np.zeros(cfg.n_classes)

        for line in image_list:

            line = line.strip()
            print(line)
            line_label = int(line.split(';')[1])

            if counter[line_label] < cfg.resized_img_num * 0.9:
                with open('train.txt', 'a') as f:
                    f.write(line + '\n')
                counter[line_label] += 1
            else:
                with open('test.txt', 'a') as f:
                    f.write(line + '\n')
                counter[line_label] += 1

        print('complete')


if __name__ == '__main__':

    counting_classes(cfg.train_dir)
    # bd = BalancedData()
    # bd.create_config_dict()
    # bd.create_resized_images()
    # bd.create_text_files()

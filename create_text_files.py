import os
import glob


class FileCreator:

    def __init__(self):

        self.dirs = {
            'train': ['/home/roman/myFolder/task6/my_drisna/train_dataset/Training_1',
                      '/home/roman/myFolder/task6/my_drisna/train_dataset/Training_2',
                      '/home/roman/myFolder/task6/my_drisna/test_dataset/Testing_1'],

            'test':  ['/home/roman/myFolder/task6/my_drisna/test_dataset/Testing_1']
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

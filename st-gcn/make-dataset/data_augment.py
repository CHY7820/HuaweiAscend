import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='./', help='data root, end with /')
parser.add_argument('-c', '--class_name', help='class name of the data')
parser.add_argument('-n', '--data_num', default=18, type=int,
                    help='max file number of a class of data')
parser.add_argument('-l', '--frame_len', default=30, type=int,
                    help='frame number of a data contains')
# parser.add_argument('-t', '--aug_type', default='all',
# help='type to augment the data: [shift, rot,all]')
# parser.add_argument('-s', '--start_num', default=None,
# help='continue to augment, result saved after this file')
# parser.add_argument('--tx', default=100,
# help='shift pixels along x axis')
# parser.add_argument('--ty', default=0,
# help='shift pixels along y axis')


class AugPara(object):
    """docstring for AugPara"""

    def __init__(self):
        super(AugPara, self).__init__()
        self.shift = [[-100, 0], [100, 0], [0, 20], [0, -20],
                      [-200, 0], [200, 0], [100, 20], [100, -20],
                      [-100, 20], [-100, -20], [200, 20], [200, -20],
                      [-200, 20], [-200, -20]]
        self.rot = [0, 5, -5]


def augment(args, aug_para):
    print(f'augment class: {args.class_name}')
    aug_num = 1
    cols = 1280
    rows = 720

    for data_id in range(1, args.data_num + 1):
        aug_num = 1
        root = args.root + f'{args.class_name}/{data_id}/'
        for s in aug_para.shift:
            for rot in aug_para.rot:
                new_root = args.root + \
                    f'{args.class_name}/{data_id+args.data_num*aug_num}/'
                if not os.path.exists(new_root):
                    os.mkdir(new_root)
                M1 = np.float32([[1, 0, s[0]], [0, 1, s[1]]])
                M2 = cv2.getRotationMatrix2D((cols / 2., rows / 2.), rot, 1)
                print(f'shift: x~{s[0]}, y~{s[1]}')
                print(f'rotate: angle~{rot}')

                for i in range(args.frame_len):

                    img = cv2.imread(root + f'{i}.jpg')
                    rows, cols = img.shape[:2]
                    img_new = cv2.warpAffine(img, M1, (cols, rows))
                    img_new = cv2.warpAffine(img_new, M2, (cols, rows))
                    cv2.imwrite(new_root + f'{i}.jpg', img_new)

                aug_num += 1

    print('- done.')


if __name__ == '__main__':
    aug_para = AugPara()
    args = parser.parse_args()

    augment(args, aug_para)

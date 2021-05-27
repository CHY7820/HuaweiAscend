import os
import argparse
import glob

parser = argparse.ArgumentParser()

parser.add_argument('--root', default='./DIR-dataset/',
                    help='file root, end with /')
parser.add_argument('-c', '--class_name', default='down',
                    help='class name of the data')


def remove(args, label):
    print('start removing.')
    txt_id = label
    step = 18
    while True:
        path = args.root + f'{args.class_name}/motion_data_{txt_id}.txt'
        if os.path.exists(path):
            os.remove(path)
            print(path + ' removed.')
        else:
            break
        txt_id += step
    print('- done.')


def sorter(item):
    # print(item)
    # print(int(item.split('/')[-1].split('.')[0].split('_')[-1]))
    return int(item.split('/')[-1].split('.')[0].split('_')[-1])


def rename(args):
    for data_id in range(1, args.data_num + 1):
        path = args.root + f'{args.class_name}/{data_id}'
        imgs = glob.glob(path + '/*.jpg')
        old_names = sorted(imgs, key=sorter)
        k = 0
        for old_name in old_names:
            new_name = path + f'/{k}.jpg'
            if new_name != old_name:
                os.rename(old_name, new_name)
                print(old_name + ' is renamed as ' + new_name)
            k += 1


def rename(args):
    print('start renaming.')
    path = args.root + args.class_name
    files = glob.glob(path + '/*.txt')
    old_names = sorted(files, key=sorter)
    k = 1
    for old_name in old_names:
        new_name = path + f'/motion_data_{k}.txt'
        if new_name != old_name:
            os.rename(old_name, new_name)
            print(old_name + ' is renamed as ' + new_name)
        k += 1

    print('- done.')


def test(args):
    left_labels = [10]
    right_labels = [9, 10, 11, 13, 17]
    up_labels = [9, 10, 11, 13, 14, 17, 18]
    down_labels = [18]
    args.class_name = 'left'

    for label in left_labels:
        remove(args, label)


if __name__ == '__main__':
    args = parser.parse_args()
    args.class_name = 'right'

    rename(args)
    # test(args)

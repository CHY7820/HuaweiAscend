import os
import argparse
import glob

parser = argparse.ArgumentParser()

parser.add_argument('--root', default='./',
                    help='rename file root, end with /')
parser.add_argument('-c', '--class_name', default='down',
                    help='class name of the data')
parser.add_argument('-n', '--data_num', default=30, type=int,
                    help='max file number of a class of data')


def sorter(item):
    # print(item)
    # print(int(item.split('/')[-1].split('.')[0]))
    return int(item.split('/')[-1].split('.')[0])


def rename(args):
    print('start renaming.')
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
    print('- done.')


def test(args):
    args.root = './data-origin-cut/'
    args.class_name = 'down'
    for data_id in range(1, args.data_num + 1):
        path = args.root + f'{args.class_name}/{data_id}'
        imgs = glob.glob(path + '/*.jpg')
        print(imgs)
        print(sorted(imgs))
        break


if __name__ == '__main__':
    args = parser.parse_args()
    rename(args)

    # test(args)

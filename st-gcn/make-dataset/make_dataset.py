import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--class_name', help='class name of the data')
parser.add_argument('-n', '--data_num', default=30, type=int,
                    help='max number of a class of data')
parser.add_argument('-l', '--frame_len', default=100, type=int,
                    help='frame number of a data contains')
parser.add_argument('-s', '--restart_frame', default=None, type=int,
                    help='file number to restart')


def make(args):
    width = 1280
    height = 720
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    i = 0
    data_id = 1
    if args.restart_frame is not None:
        data_id = args.restart_frame

    print('start making data.')
    while True:
        print(f'class: {args.class_name}, id: {data_id}, num: {i}')
        if i == args.frame_len:
            i = 0
            data_id += 1
            print("==================NEXT==================")
        if i == 0:
            path = f'./{args.class_name}/{data_id}'
            if not os.path.exists(path):
                os.mkdir(path)

        ret, frame = cap.read()
        cv2.imwrite(f'./{args.class_name}/{data_id}/{i}.jpg', frame)
        i += 1

        if data_id == args.data_num:
            break
    print('- done')
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.class_name):
        os.mkdir(args.class_name)
    make(args)

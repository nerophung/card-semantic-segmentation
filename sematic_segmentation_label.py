import argparse
import cv2
import imutils
import numpy as np
import os

args = argparse.ArgumentParser()
args.add_argument('-image_dir', '--image_dir', required=True)
args.add_argument('-log', '--log_dir', required=True)
args.add_argument('-begin_index', '--begin_index', default=0, type=int)

list_points = []

def mark_point(event, x, y, flags, param):
    list_points = param[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        list_points.append([x, y])
        draw_points = np.array([list_points], np.int32)
        draw_image = resize_image.copy()
        cv2.polylines(draw_image, [draw_points], True, (0, 255, 0), thickness=2)
        cv2.imshow('Image', draw_image)

if __name__ == '__main__':
    args = vars(args.parse_args())
    image_dir = args['image_dir']
    log_dir = args['log_dir']
    index = args['begin_index']
    train_dir = os.path.join(log_dir, 'train')
    label_dir = os.path.join(log_dir, 'label')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    images = os.listdir(image_dir)
    for image_name in images:
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        original_height, original_width, _ = image.shape
        resize_image = imutils.resize(image, height=1000)
        label_image = np.zeros(resize_image.shape)
        list_points = []
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', mark_point, [list_points])
        cv2.imshow('Image', resize_image)
        key = cv2.waitKey(0)
        while key != 13:
            if key == ord('z'):
                del list_points[-1]
                draw_image = resize_image.copy()
                draw_points = np.array([list_points], np.int32)
                cv2.polylines(draw_image, [draw_points], False, (0, 255, 0), thickness=2)
                cv2.imshow('Image', draw_image)
                key = cv2.waitKey(0)
        label_image = cv2.fillPoly(label_image, np.array([list_points]), (255, 255 ,255))
        # Save numpy array file
        label_image = label_image[:, :, 0]
        label_image = cv2.resize(label_image, (256, 256))
        image = cv2.resize(image, (256, 256))
        _, label_image = cv2.threshold(label_image, 127, 255, cv2.THRESH_BINARY)
        label_image = np.array(label_image, dtype=np.uint8)
        npy_train_path = os.path.join(train_dir, 'train_{}.npy'.format(index))
        npy_label_path = os.path.join(label_dir, 'label_{}.npy'.format(index))
        np.save(npy_train_path, image)
        np.save(npy_label_path, label_image)
        index+= 1



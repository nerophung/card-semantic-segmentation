import argparse
import cv2
import imutils
import numpy as np
import os

args = argparse.ArgumentParser()
args.add_argument('-image_dir', '--image_dir', required=True)
args.add_argument('-log', '--log_dir', required=True)

list_points = []

def mark_point(event, x, y, flags, param):
    global list_points
    if event == cv2.EVENT_LBUTTONDOWN:
        list_points.append((x, y))
        cv2.circle(resize_image, (x, y), 2, (0, 255, 0), -1)
        cv2.imshow('Image', resize_image)
        print((x, y))


def order_points(pts):
    pts = np.array(pts, dtype='float32')
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = np.array(pts, dtype='float32')
    tl, tr, br, bl = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

if __name__ == '__main__':
    args = vars(args.parse_args())
    image_dir = args['image_dir']
    log_dir = args['log_dir']
    images = os.listdir(image_dir)
    for image in images:
        image_path = os.path.join(image_dir, image)
        image = cv2.imread(image_path)
        original_height, original_width, _ = image.shape
        resize_image = imutils.resize(image, height=1000)
        height, width, _ = resize_image.shape
        r_x = original_width/width
        r_y = original_height/height
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', mark_point)
        cv2.imshow('Image', resize_image)
        cv2.waitKey(0)
        new_list_point = []
        for point in list_points:
            x, y = point
            x = int(x * r_x)
            y = int(y* r_y)
            new_list_point.append([x, y])
        image = four_point_transform(image, new_list_point)
        cv2.imwrite('logs/{}.png'.format(os.path.basename(image_path)), image)
        list_points = []

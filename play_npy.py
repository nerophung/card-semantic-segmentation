import numpy as np
import cv2
import argparse

args = argparse.ArgumentParser()
args.add_argument('-npy_file', '--npy_file', required=True)

list_points = []

if __name__ == '__main__':
    args = vars(args.parse_args())
    image = np.load(args['npy_file'])
    cv2.imshow('Image', image)
    cv2.waitKey(0)


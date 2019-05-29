import cv2
import numpy as np
from pathlib import Path


def get_images(file_name: str):
    with open(file_name, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder="big")
        num_img = int.from_bytes(f.read(4), byteorder="big")
        rows = int.from_bytes(f.read(4), byteorder="big")
        cols = int.from_bytes(f.read(4), byteorder="big")

        if magic_number == 2051:
            images = list()
            for _i in range(num_img):
                images.append(np.array([[int.from_bytes(f.read(1), byteorder="big")
                                         for _c in range(cols)] for _r in range(rows)], dtype=np.uint8))
            return images
        else:
            raise ValueError("Unknown Magic Number: {}".format(magic_number))


def get_labels(file_name: str):
    with open(file_name, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder="big")
        num_items = int.from_bytes(f.read(4), byteorder="big")

        if magic_number == 2049:
            return [int.from_bytes(f.read(1), byteorder="big") for _ in range(num_items)]
        else:
            raise ValueError("Unknown Magic Number: {}".format(magic_number))


def save_images(images, base_dir: str, base_name: str = "img_"):
    num_digits = len(str(len(images)))
    save_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    for n, i in enumerate(images):
        cv2.imwrite("{}{}{}.png".format(base_dir, base_name, str(n).zfill(num_digits)), i, save_params)


def read_images(base_dir: str, base_name: str = "img_", read_limit: int = -1):
    p_list = Path(base_dir).glob('**/{}*.png'.format(base_name))
    p_list = sorted(p_list, key=lambda p: str(p))
    if read_limit <= -1:
        return [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in p_list]
    else:
        images = list()
        for p in p_list:
            if len(images) == read_limit:
                break
            images.append(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
        return images

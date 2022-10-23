#
# Copyright (c) 2016,2018-2019 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
import argparse
import numpy as np
import os

from PIL import Image

RESIZE_METHOD_ANTIALIAS = "antialias"
RESIZE_METHOD_BILINEAR  = "bilinear"

def __resize_square_to_jpg(src, dst, size,resize_type):
    src_img = Image.open(src)
    # If black and white image, convert to rgb (all 3 channels the same)
    if len(np.shape(src_img)) == 2: src_img = src_img.convert(mode = 'RGB')
    # center crop to square
    width, height = src_img.size
    short_dim = min(height, width)
    crop_coord = (
        (width - short_dim) / 2,
        (height - short_dim) / 2,
        (width + short_dim) / 2,
        (height + short_dim) / 2
    )
    img = src_img.crop(crop_coord)
    # resize to inceptionv3 size
    if resize_type == RESIZE_METHOD_BILINEAR :
        dst_img = img.resize((size, size), Image.BILINEAR)
    else :
        dst_img = img.resize((size, size), Image.ANTIALIAS)
    # save output - save determined from file extension
    dst_img.save(dst)
    return 0

def convert_img(src,dest,size,resize_type):
    print("Converting images for inception v3 network.")

    print("Scaling to square: " + src)
    for root,dirs,files in os.walk(src):
        for jpgs in files:
            src_image=os.path.join(root, jpgs)
            if('.png' in src_image or '.jpg' in src_image):
                print(src_image)
                dest_image = os.path.join(dest, jpgs[:-3]+"jpg")
                __resize_square_to_jpg(src_image,dest_image,size,resize_type)

def main():
    parser = argparse.ArgumentParser(description="Batch convert jpgs",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dest',type=str, required=True)
    parser.add_argument('-s','--size',type=int, default=299)
    parser.add_argument('-i','--img_folder',type=str, required=True)
    parser.add_argument('-r','--resize_type',type=str, default=RESIZE_METHOD_BILINEAR,
                        help='Select image resize type antialias or bilinear. Image resize type should match '
                             'resize type used on images with which model was trained, otherwise there may be impact '
                             'on model accuracy measurement.')

    args = parser.parse_args()

    size = args.size
    src = os.path.abspath(args.img_folder)
    dest = os.path.abspath(args.dest)
    resize_type = args.resize_type

    assert resize_type == RESIZE_METHOD_BILINEAR or resize_type == RESIZE_METHOD_ANTIALIAS, \
           "Image resize method should be antialias or bilinear"

    convert_img(src,dest,size,resize_type)

if __name__ == '__main__':
    exit(main())

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import types
from pylsd.lsd import lsd
import glob
import json
from natsort import natsorted

#自作関数群
import edge_color
import linlin
import crop



if __name__ == '__main__':
    f = open('set.json', 'r', encoding="utf-8")
    json_data = json.load(f)
    piece = json_data["piece"]
    resultpath = json_data["peice_res"]
    Complete = json_data["Complete"]
    matching = json_data["matching"]
    line = json_data["line"]
    IMAGES_PATH = natsorted(glob.glob(piece + "*"))
    print(IMAGES_PATH)
    namecount = 1

    pictcount = len(IMAGES_PATH)
    for pict in IMAGES_PATH:
        alpha_img = crop.crop(pict)
        cv2.imwrite(resultpath + str(namecount) + ".png",alpha_img)
        namecount = namecount +1
        if pictcount < namecount:
            namecount = 1

    alpha_path = natsorted(glob.glob(resultpath + "*"))
    print()
    for res in alpha_path:

        line_res = linlin.liner(res)
        cv2.imwrite(line +str(namecount) + ".png",line_res)

        sift = edge_color.edge_color(res,Complete)
        cv2.imwrite(matching + str(namecount) + ".png",sift)
        namecount = namecount +1
        if pictcount < namecount:
            namecount = 1
        

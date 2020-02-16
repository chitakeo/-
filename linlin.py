import cv2
import sys
import numpy as np
import types
from pylsd.lsd import lsd
import glob

def liner(img_path):
    #ハフ変換による直線検出                                       
    img = cv2.imread(img_path)
    img = cv2.resize(img,(int(img.shape[1]/5),int(img.shape[0]/5)))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),5)

    edges = cv2.Canny(gray,50,150,apertureSize = 3)

##################
###ピースの分類###
##################

    linesH = cv2.HoughLinesP(edges, rho=1, theta=np.pi/360, threshold=80, minLineLength=10, maxLineGap=60)
    if type(linesH) == type(None):
        linesH = []
    
    if len(linesH) == 2:
        print("角のピース")
        print(len(linesH),"lines")
    elif len(linesH) != 2:
        linesH = []
        linesH = cv2.HoughLinesP(edges, rho=1, theta=np.pi/360, threshold=50, minLineLength=10, maxLineGap=60)

        if type(linesH) == type(None):
            linesH = []
        if len(linesH) == 1:
            print("四辺のピース")
            print(len(linesH),"lines")
        elif len(linesH) != 1:
            lines = []
            linesH = cv2.HoughLinesP(edges, rho=1, theta=np.pi/360, threshold=100, minLineLength=10, maxLineGap=60)

            if type(linesH) == type(None):
                linesH = []
            if len(linesH) == 1:
                print("四辺のピース")
                print(len(linesH),"lines")
                
            elif len(linesH) == 0:
                print("中央のピース")
                print(len(linesH),"lines")
                #引く直線がないので終了
                #sys.exit() 
                pass
            
    img2 = img.copy()
    for line in linesH:
        x1, y1, x2, y2 = line[0]
        # 赤線を引く                                                                   
        img2 = cv2.line(img2, (x1,y1), (x2,y2), (0,0,255), 3)

    return img2

if __name__ == '__main__':
    IMAGES_PATH = glob.glob('/Users/tuhadaiki/画像処理実験/cutmask/pict/*')
    namecount = 1
    for pict in IMAGES_PATH:
        res = liner(pict)
        cv2.imwrite("/Users/tuhadaiki/画像処理実験/cutmask/line_result/" + str(namecount) + ".png" , res)
        namecount = namecount +1
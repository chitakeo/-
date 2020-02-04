import cv2
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np

def crop():

    IMAGE_PATH = "/Users/e175729/engineering_design_exercise/pict/piece19.JPG"
    # 画像を読み込む。
    src = cv2.imread(IMAGE_PATH)
    # 2値化する。ここで画像によってうまく調整
    dst = cv2.inRange(src, (120, 120, 120), (255, 255, 255))
    tmp_img, contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    n = 0
    maxper = 0
    secper = 0
    maxn = 0
    secn = 0
    #取得できた輪郭のうち一番大きい面積を探索
    while len(contours) > n:
        cnt = contours[n]
        perimeter = cv2.contourArea(cnt,True)
        if maxper < perimeter:
            secper = maxper
            maxper = perimeter
            secn = maxn
            maxn = n
        elif secper < perimeter:
            secper = perimeter
            secn = n

        n+=1

    cnt = contours[maxn]
    area = cv2.contourArea(cnt,True)
    #取得できた輪郭の内側の面積と画像サイズの比
    #print((area/src.size)*100)

    #取得できた輪郭の内側の面積と画像サイズの比によって処理を変更
    if ((area/src.size)*100 > 1):
        external_contours = np.copy(src)
        external_contours = np.zeros(src.shape)

        #マスク用画像作成
        if hierarchy[0][maxn][3] == -1:

            cv2.drawContours(external_contours, contours, maxn, 255, -1)
            cv2.imwrite("for_musk.jpg",external_contours)

        img = cv2.drawContours(src, contours, maxn, (0, 255, 0), 3)

        # load image, change color spaces, and smoothing
        org_img2 = cv2.imread(IMAGE_PATH)
        # Reading the original image
        org_img = cv2.imread("for_musk.jpg")
        hsv = cv2.cvtColor(org_img, cv2.COLOR_BGR2HSV)
        # Spliting to H,S,V
        h_img, s_img, v_img = cv2.split(hsv)
        _thre, img_flowers = cv2.threshold(s_img, 100, 255, cv2.THRESH_BINARY)

        s_img = cv2.bitwise_not(img_flowers)

        # Flattening a histgram of s_img
        hist_s_img = cv2.equalizeHist(img_flowers)

        # Binarization
        _, result_bin = cv2.threshold(
        s_img, 200, 255, cv2.THRESH_BINARY)

        # Morphing（Closing）
        ## Setting filters
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
        ## Execution morphing
        result_morphing = cv2.morphologyEx(
        result_bin, cv2.MORPH_CLOSE, kernel)


        mask = cv2.bitwise_not(result_morphing)#白黒反転させて、mask画像を作る。

        dst = cv2.medianBlur(mask, ksize=37)#光の反射などで、一部なくなってしまうので、それを補完する。

        result = cv2.bitwise_and(org_img2, org_img2, mask=mask) # 元画像とマスクを合成

        # Detection contours
        ##########
        #そのままではピースの周りが全部黒いままのjpgが出力されるので、透過画像に変換する。
        img_bgr = cv2.split(result)#b,g,rの値をとる。
        mask2 = np.where((dst==2)|(dst==0),0,1).astype('uint8')#maskとして使うdstの値を参照。
        mask2 = mask2*255#alpha値作成

        # cv2.imwrite("aaa.png",mask2*255)

        img_alpha = cv2.merge(img_bgr+[mask2])#黒枠の透過画像生成

        cv2.imwrite("crop.png",img_alpha)#出力。透過画像は.png形式でないと出力できない。
        # Setting for display
    else:
        # load image, change color spaces, and smoothing
        img = cv2.imread(IMAGE_PATH)
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #retval, img_mono = cv2.threshold(grayImg, threshhold, 255, cv2.THRESH_BINARY)
        #cv2.imwrite("mono.jpg",img_otsu)
        # detect tulips
        img_H, img_S, img_V = cv2.split(img_HSV)
        cv2.imwrite("H1.jpg",img_H)
        cv2.imwrite("S1.jpg",img_S)
        cv2.imwrite("V1.jpg",img_V)
        _thre, img_flowers = cv2.threshold(img_S, 50, 255, cv2.THRESH_BINARY)
        cv2.imwrite("S2.jpg",img_flowers)
        tmp_img, contours, hierarchy = cv2.findContours(img_flowers, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Contour approximation
        approx = approx_contour(contours)

        # Contour line drawing
        cp_org_img_for_draw = np.copy(img)
        cp_org_img_for_draw2 = np.copy(img)
        drawing_edge(img, approx, cp_org_img_for_draw,cp_org_img_for_draw2,hierarchy)

        org_img2 = cv2.imread(IMAGE_PATH)
        # Reading the original image
        org_img = cv2.imread("for_musk.jpg")
        hsv = cv2.cvtColor(org_img, cv2.COLOR_BGR2HSV)
        # Spliting to H,S,V
        h_img, s_img, v_img = cv2.split(hsv)
        _thre, img_flowers = cv2.threshold(s_img, 100, 255, cv2.THRESH_BINARY)

        s_img = cv2.bitwise_not(img_flowers)

        # Flattening a histgram of s_img
        hist_s_img = cv2.equalizeHist(img_flowers)

        # Binarization
        _, result_bin = cv2.threshold(
            s_img, 200, 255, cv2.THRESH_BINARY)

        # Morphing（Closing）
        ## Setting filters
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
        ## Execution morphing
        result_morphing = cv2.morphologyEx(
            result_bin, cv2.MORPH_CLOSE, kernel)


        mask = cv2.bitwise_not(result_morphing)#白黒反転させて、mask画像を作る。

        dst = cv2.medianBlur(mask, ksize=37)#光の反射などで、一部なくなってしまうので、それを補完する。

        result = cv2.bitwise_and(org_img2, org_img2, mask=mask) # 元画像とマスクを合成

        # Detection contours
    ##########
        #そのままではピースの周りが全部黒いままのjpgが出力されるので、透過画像に変換する。
        img_bgr = cv2.split(result)#b,g,rの値をとる。
        mask2 = np.where((dst==2)|(dst==0),0,1).astype('uint8')#maskとして使うdstの値を参照。
        mask2 = mask2*255#alpha値作成

        # cv2.imwrite("aaa.png",mask2*255)

        img_alpha = cv2.merge(img_bgr+[mask2])#黒枠の透過画像生成

        cv2.imwrite("crop.png",img_alpha)#出力。透過画像は.png形式でないと出力できない。
        # Setting for display

        # Execution display
        #display_result(
            #org_img, h_img, s_img, v_img, hist_s_img,
        #    result_bin, result_morphing, cp_org_img_for_draw)

def approx_contour(contours):
        ######################################################
        # 輪郭直線近似
        ######################################################
    approx = []
    for i in range(len(contours)):
        cnt = contours[i]
        epsilon = 0.0001*cv2.arcLength(cnt,True)
        approx.append(cv2.approxPolyDP(cnt,epsilon,True))
    return approx


def drawing_edge(org_img, contours, cp_org_img_for_draw,cp_org_img_for_draw2,hierarchy):
    ######################################################
    # 輪郭線描画
    ######################################################
    min_area = 100
    large_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > min_area]



    n = 0
    maxper = 0
    secper = 0
    maxn = 0
    secn = 0
    while len(large_contours) > n:
        cnt = large_contours[n]
        perimeter = cv2.arcLength(cnt,True)
        if maxper < perimeter:
            secper = maxper
            maxper = perimeter
            secn = maxn
            maxn = n
        elif secper < perimeter:
            secper = perimeter
            secn = n

        n+=1

    external_contours = np.copy(cp_org_img_for_draw)
    cv2.drawContours(cp_org_img_for_draw, large_contours, secn, (0, 255, 0), 3)
    cv2.imwrite("seccor.jpg",cp_org_img_for_draw)
    cv2.drawContours(cp_org_img_for_draw2, large_contours, maxn, (0, 255, 0), 3)
    cv2.imwrite("allcor.jpg",cp_org_img_for_draw2)
    external_contours = np.zeros(cp_org_img_for_draw.shape)

    #輪郭の内側を青で塗りつぶす
    if hierarchy[0][maxn][3] == -1:

        cv2.drawContours(external_contours, large_contours, maxn, 255, -1)
        cv2.imwrite("for_musk.jpg",external_contours)

    plt.imshow(external_contours,cmap='gray')


if __name__ == '__main__':
    crop()

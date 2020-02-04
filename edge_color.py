import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt



def main():
    img_piece = cv2.imread("kirinuki.png") # ファイル読み込み
    img_zentai = cv2.imread("zentai.jpg")

    #画像の平滑化
    blur_piece = blur(img_piece)
    blur_zentai = blur(img_zentai)
    
    #輪郭の抽出
    edges_piece = make_edge(blur_piece)
    edges_zentai = make_edge(blur_zentai)

    # BGRでの色抽出
    bgrLower = np.array([70,70,70])    # 抽出する色の下限
    bgrUpper = np.array([255, 255, 255])    # 抽出する色の上限
    bgrResult_piece = bgrExtraction(img_piece, bgrLower, bgrUpper)
    bgrResult_zentai = bgrExtraction(img_zentai, bgrLower, bgrUpper)
    
    #クロージング処理
    kernel = np.ones((9,9),np.uint8)
    closing_piece = cv2.morphologyEx(bgrResult_piece, cv2.MORPH_CLOSE, kernel)
    closing_zentai = cv2.morphologyEx(bgrResult_zentai, cv2.MORPH_CLOSE, kernel)
    
    #SIFTによるマッチング
    #エッジ
    sift_matching = matching(edges_piece, edges_zentai)
    
    #色
    sift_matching = matching(bgrResult_piece, bgrResult_zentai)
    
    
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow('image', sift_matching)
    #sleep(1)
    while True:
        # キー入力を1ms待って、keyが「q」だったらbreak
        key = cv2.waitKey(1)&0xff
        if key == ord('q'):
            break

    cv2.destroyAllWindows()





## 関数定義 ##

# BGRで特定の色を抽出する関数
def bgrExtraction(image, bgrLower, bgrUpper):
    img_mask = cv2.inRange(image, bgrLower, bgrUpper) # BGRからマスクを作成
    result = cv2.bitwise_and(image, image, mask=img_mask) # 元画像とマスクを合成
    return result
    
# エッジを抽出する関数
def make_edge(image):
    med_val = np.median(image)
    lower = int(max(0, 0.15* med_val))
    upper = int(min(255, 0.65* med_val))
    result = cv2.Canny(image, threshold1 = lower, threshold2 = upper)
    return result
    
#画像の平滑化をする関数
def blur(image):
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	result = cv2.blur(gray,ksize = (3,3))
	return result


def matching(image1, image2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1,None)
    kp2, des2 = sift.detectAndCompute(image2,None)
    bf = cv2.BFMatcher()
    matchers = bf.knnMatch(des1, des2, k = 2)
    good = []
    for match1, match2 in matchers:
        if match1.distance < 0.5 * match2.distance:
            good.append([match1])
    image3 = cv2.drawMatchesKnn(image1, kp1, image2, kp2, good, None, flags = 2)
    return image3
    



if __name__ == '__main__':
    main()
"""
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow('image', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
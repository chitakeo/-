import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

#ピースのBGRそれぞれの中央値または、平均値を取りSIFTでマッチング
#結果としてはピースと全体の写真のそれぞれの撮影時の光の入り方が異なるため期待するような結果は出ない
#特定の色領域に絞ることで、その色部分とのマッチングができたら嬉しいなっていう発想

def edge_color(piece,Complete):
    img_piece = cv2.imread(piece) # ファイル読み込み
    img_zentai = cv2.imread(Complete)
    
    # 色による判別時使用
    # average = 0, median = any
    ave_med = 0
    # 画像の中央値または平均値にかける値
    # low = 0.3 upp = 1.5 くらいがちょうどいいかも 
    low = 0.3
    upp = 1.5

    #画像の平滑化
    blur_piece = blur(img_piece)
    blur_zentai = blur(img_zentai)
    
    #輪郭の抽出
    edges_piece = make_edge(blur_piece)
    edges_zentai = make_edge(blur_zentai)
    
    # 対象範囲を切り出し
    height, width, channels = img_piece.shape[:3]
    boxFromX = 0 #対象範囲開始位置 X座標
    boxFromY = 0 #対象範囲開始位置 Y座標
    boxToX = width #対象範囲終了位置 X座標
    boxToY = height #対象範囲終了位置 Y座標
    # y:y+h, x:x+w　の順で設定
    imgBox = img_piece[boxFromY: boxToY, boxFromX: boxToX]

    # ピース以外の部分を削除
    # 切り取り後の画像を使用予定のため,BGRの値が0の場合は削除
    b_img = imgBox.T[0][imgBox.T[0] > 0]
    g_img = imgBox.T[1][imgBox.T[1] > 0]
    r_img = imgBox.T[2][imgBox.T[2] > 0]

    # RGB平均値を出力
    # flattenで一次元化しmeanで平均を取得 
    # print(imgBox.T[0])
    b_flat = b_img.flatten()
    g_flat = g_img.flatten()
    r_flat = r_img.flatten()
   
   # 平均値を取るか、中央値を取るのか
    if ave_med  == 0:
        b = b_flat.mean()
        g = g_flat.mean()
        r = r_flat.mean()
    else:
        b = np.median(b_flat)
        g = np.median(g_flat)
        r = np.median(r_flat)
        
    #色抽出の下限と上限を指定
    b_lower = int(max(0, low * b))
    g_lower = int(max(0, low * g))
    r_lower = int(max(0, low * r))

    b_upper = int(min(255, upp * b))
    g_upper = int(min(255, upp * g))
    r_upper = int(min(255, upp * r))
    

    # BGRでの色抽出
    bgrLower = np.array([b_lower, g_lower, r_lower])    # 抽出する色の下限
    bgrUpper = np.array([b_upper, g_upper, r_upper])    # 抽出する色の上限
    bgrResult_piece = bgrExtraction(img_piece, bgrLower, bgrUpper)
    bgrResult_zentai = bgrExtraction(img_zentai, bgrLower, bgrUpper)
    
    #クロージング処理
    kernel = np.ones((9,9),np.uint8)
    closing_piece = cv2.morphologyEx(bgrResult_piece, cv2.MORPH_CLOSE, kernel)
    closing_zentai = cv2.morphologyEx(bgrResult_zentai, cv2.MORPH_CLOSE, kernel)
    
    #SIFTによるマッチング
    #エッジ
    #sift_matching = matching(edges_piece, edges_zentai)
    
    #色
    sift_matching = matching(bgrResult_piece, bgrResult_zentai)
    return sift_matching
    """
    cv2.imwrite("test.png",sift_matching)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow('image', sift_matching)
    #sleep(1)

    while True:
        # キー入力を1ms待って、keyが「q」だったらbreak
        key = cv2.waitKey(1)&0xff
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    """






## 関数定義 ##

# BGRで閾値を設定し抽出する関数
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

# SIFTを用いたマッチング
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
   piece =  ("/Users/tuhadaiki/画像処理実験/imgfile/piece_img/pict/piece2.JPG")
   Complete =("/Users/tuhadaiki/画像処理実験/imgfile/Complete_puzzle/kumonma.png")
   edge_color(piece,Complete)
   cv2.imwrite("sift.png",sift_matching)
"""
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow('image', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
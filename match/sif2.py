import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


kan_file_path = 'save/kuma.jpg'   #完成図保存のファイルパス&ファイル名
mach_file_path = 'save/mach_kuma.jpg' #マッチング結果保存のファイルパス&ファイル名
pazu_file_path = 'pazu/kousiki.png'      #パズルのファイルパス&ファイル名
pice_file_path = 'pazu/rb.png'   #ピースのファイルパス&ファイル名

img1 = cv2.imread(pazu_file_path)          # 元画像
img2 = cv2.imread(pice_file_path)    # パズルのピース
height, weight = img1.shape[:2]
height2, weight2 = img2.shape[:2]
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


sift = cv2.xfeatures2d.SIFT_create()


kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)


bf = cv2.BFMatcher()
matches =bf.knnMatch(des1,des2, k=2)


good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])


good = sorted(good, key = lambda x:x[0].distance)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[:5],None,flags=2)
cv2.imwrite(mach_file_path, img3)
cv2.imwrite(mach_file_path, img3)
img1_pt = [list(map(int, kp1[m[0].queryIdx].pt)) for m in good]
img2_pt = [list(map(int, kp2[m[0].trainIdx].pt)) for m in good]

###
###
#ここからマッチングがうまくいったと仮定している
#元画像の縦と横の距離を計算
img1_hei = img1_pt[0][1] - img1_pt[1][1]
if img1_hei < 0:
	img1_hei = img1_hei * -1

img1_wei = img1_pt[0][0] - img1_pt[1][0]
if img1_wei < 0:
	img1_wei = img1_wei * -1

#ピースの縦と横の距離を計算
img2_hei = img2_pt[0][1] - img2_pt[1][1]
if img2_hei < 0:
	img2_hei = img2_hei * -1

img2_wei = img2_pt[0][0] - img2_pt[1][0]
if img2_wei < 0:
	img2_wei = img2_wei * -1




#ピースと元画像の縦と横の比率を計算
if img1_hei == 0 and img2_hei  == 0:
	if img1_wei == 0 and img2_wei == 0:
		pass
	else:
		wei = img2_wei / img1_wei
		img2 = cv2.resize(img2, (int(weight2/wei), int(height2)))
elif img1_wei == 0 and img2_wei == 0:
	hei = img2_hei / img1_hei
	img2 = cv2.resize(img2, (int(weight2), int(height2/hei)))
elif img1_pt[0][1] != img2_pt[0][1] and img1_pt[0][0] != img2_pt[0][0]:
	wei = img2_wei / img1_wei
	hei = img2_hei / img1_hei
	img2 = cv2.resize(img2, (int(weight2/wei), int(height2/hei)))



gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#リサイズした画像とまたマッチング
kp2, des2 = sift.detectAndCompute(gray2,None)
matches =bf.knnMatch(des1,des2, k=2)
mac = []

for m,n in matches:
    if m.distance < 0.75*n.distance:
        mac.append([m])


mac = sorted(mac, key = lambda x:x[0].distance)
img4 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,mac[:5],None,flags=2)
img4_pt = [list(map(int, kp1[m[0].queryIdx].pt)) for m in mac]
img3_pt = [list(map(int, kp2[m[0].trainIdx].pt)) for m in mac]

#新しいピースの大きさ取得
height2,weight2 = img2.shape[:2]

print(img4_pt[0][0])
print(img1_pt[0][0])

#右下の座標からの距離を計算
hei4 = height2 - img3_pt[0][1]
wei4 = weight2 - img3_pt[0][0]


h1 = img4_pt[0][1] - img3_pt[0][1]
h2 = img4_pt[0][1] + hei4
w1 = img4_pt[0][0] - img3_pt[0][0]
w2 = img4_pt[0][0] + wei4 


if os.path.isfile(kan_file_path) :
	imageArray = cv2.imread(kan_file_path) 
else:
	imageArray = np.ones((height, weight, 3), np.uint8)*255


imageArray[h1 : h2 , w1 : w2 ] = img2

dst = cv2.addWeighted(img1, 0.5, imageArray, 0.5, 0)
cv2.imwrite(kan_file_path, imageArray)

while(1):
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.imshow('image',dst)
	if cv2.waitKey(20) & 0xFF == 27:
		break
cv2.destroyAllWindows()
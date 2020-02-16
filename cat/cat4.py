import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


kan_file_path = '../save/fe.jpg'   #完成図保存のファイルパス&ファイル名
mach_file_path = '../save/zikken.jpg' #マッチング結果保存のファイルパス&ファイル名
pazu_file_path = '../pazu/kousiki.png'      #パズルのファイルパス&ファイル名
pice_file_path = '../pazu'   #ピースのファイルパス&ファイル名
pice = ['/rt.png', '/rrb.png',  '/lt.png', '/lb.png']  #4隅の画像のファイルパス


#SIFT
def sif(gray1, gray2):
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
	return good, kp1, kp2
	

#マッチング後の組み立て
def create(img1t,img1,img2,gray1,kp1,kp2, n):
	
	img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[:5],None,flags=2)
	cv2.imwrite(mach_file_path, img3)
	img1_pt = [list(map(int, kp1[m[0].queryIdx].pt)) for m in good]
	img2_pt = [list(map(int, kp2[m[0].trainIdx].pt)) for m in good]


	#ここからマッチングがうまくいったと仮定している
	#元画像の縦と横の距離を計算
	img1_hei = img1_pt[0][1] - img1_pt[1][1]
	img1_wei = img1_pt[0][0] - img1_pt[1][0]

	#ピースの縦と横の距離を計算
	img2_hei = img2_pt[0][1] - img2_pt[1][1]
	img2_wei = img2_pt[0][0] - img2_pt[1][0]



	#ピースと元画像の縦と横の比率を計算
	height2, weight2 = img2.shape[:2]
	if img1_hei == 0 and img2_hei  == 0:
		if img1_wei == 0 and img2_wei  == 0:
			pass
		else:
			wei = abs(img2_wei) / abs(img1_wei)
			img2 = cv2.resize(img2, (int(weight2/wei), int(height2)))
	elif img1_wei == 0 and img2_wei == 0:
		hei = abs(img2_hei) / abs(img1_hei)
		img2 = cv2.resize(img2, (int(weight2), int(height2/hei)))
	elif img1_pt[0][1] != img2_pt[0][1] and img1_pt[0][0] != img2_pt[0][0]:
		wei = abs(img2_wei) / abs(img1_wei)
		hei = abs(img2_hei) / abs(img1_hei)
		img2 = cv2.resize(img2, (int(weight2/wei), int(height2/hei)))


	height, weight = img1t.shape[:2]
	#全体画像と同じ大きさの画像を作成、あるなら作らない
	if os.path.isfile(kan_file_path) :
		imageArray = cv2.imread(kan_file_path) 
	else:
		imageArray = np.ones((height, weight, 3), np.uint8)*255
	
	height2,weight2 = img2.shape[:2]
	hei41 = int(height / 4)
	hei43 = int(hei41 * 3)
	wei41 = int(weight / 4)
	wei43 = int(wei41 * 3)
	#nが0の時が左上,1の時左下、２の時右上、３の時右下
	if n == 0:
		h1 = 0
		h2 = height2
		w1 = 0
		w2 = weight2
	elif n == 1:
		h1 = height - height2
		h2 = height
		w1 = 0
		w2 = weight2
	elif n == 2:
		h1 = 0
		h2 = height2
		w1 = weight - weight2
		w2 = weight
	elif n == 3:
		h1 = height - height2
		h2 = height
		w1 = weight - weight2
		w2 = weight
	
	
		
	imageArray[h1 : h2 , w1 : w2 ] = img2
	cv2.imwrite(kan_file_path, imageArray)
	return imageArray


#4隅の画像がどこか探す
def find4(gray1):
	good = []
	max_len = 0
	for i in range(4):
		f = pice_file_path + pice[i]
		img2_1 = cv2.imread(f)
		gray2_1 =  cv2.cvtColor(img2_1, cv2.COLOR_BGR2GRAY)
		check = []
		check, kp3, kp4 = sif(gray1, gray2_1)
		print(len(check))
		if max_len < len(check):
			max_len = len(check)
			img2 = img2_1
			good = check
			kp1 = kp3
			kp2 = kp4
			
	
		
	return good, img2, kp1, kp2
	

if __name__ == '__main__':
	img1 = cv2.imread(pazu_file_path)          # 元画像
	height, weight = img1.shape[:2]
	hei41 = int(height / 4)
	hei43 = int(hei41 * 3)
	wei41 = int(weight / 4)
	wei43 = int(wei41 * 3)
	img1_lt = img1[0 : hei41 , 0 : wei41]
	img1_lb = img1[hei43 : height , 0 : wei41]
	img1_rt = img1[0 : hei41 , wei43 : weight]
	img1_rb = img1[hei43 : height , wei43 : weight]
	coner = [img1_lt, img1_lb, img1_rt, img1_rb]  #左上、左下、右上、右下の順番にして
	for n in range(4):
		img1_tk = coner[n]
		gray1 = cv2.cvtColor(img1_tk, cv2.COLOR_BGR2GRAY)
		good = []
		good, img2, kp1, kp2 = find4(gray1)
		dst = create(img1,img1_tk, img2, gray1, kp1, kp2, n)


	while(1):
		cv2.namedWindow('image', cv2.WINDOW_NORMAL)
		cv2.imshow('image',dst)
		if cv2.waitKey(20) & 0xFF == 27:
			break
	cv2.destroyAllWindows()
import numpy as np
import cv2
from matplotlib import pyplot as plt

#https://gist.github.com/kphilipp/7271334

ratio = 0.75
MIN_MATCH_COUNT = 15

img = cv2.imread('image\casa315g.jpg')   # queryImage  C:\Users\Miguel Lima\PycharmProjects\mestrado\venv\steps\image\casa315g.jpg
img2 = cv2.imread('image\Casa120.jpeg')  #Trainning Image C:\Users\Miguel Lima\PycharmProjects\mestrado\venv\steps\image\Casa120.jpeg

grayimg =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
grayimg2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

#sift = cv2.xfeatures2d.SIFT_create()
#sift = cv2.xfeatures2d.SIFT_create()
#kp = sift.detect(grayimg,None)
orb = cv2.ORB_create(nfeatures=1500, scoreType=cv2.ORB_FAST_SCORE,edgeThreshold=20, patchSize=20)

kp1, des1 = orb.detectAndCompute(grayimg,None)
#src_orb = [o.pt for o in kp1]

kp2, des2 = orb.detectAndCompute(grayimg2,None)
#src_orb_test = [o.pt for o in kp2]


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

#matches = bf.match(des1,des2)
matches = bf.knnMatch(des1,des2, k=2)

#matches = sorted(matches, key = lambda x:x.distance)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < ratio*n.distance:
        good.append(m)

print(len(matches))
print(len(kp1), len(kp2))
#print(len(src_orb), len(src_orb_test))

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h, w,_ = img.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img3 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

imgFinale = cv2.drawMatches(img,kp1,img2,kp2,good,None,**draw_params)

#matches = [m for m,mask in zip(matches,matchesMask) if mask]
#imgFinale = cv2.drawMatchesKnn(img, kp1, img2, kp2,matches,None, flags=2)

plt.imshow(imgFinale, 'gray'),plt.show()
cv2.imwrite('novaimagem.jpg',imgFinale)
cv2.imshow('img',imgFinale)
cv2.waitKey(0)
cv2.destroyAllWindows()




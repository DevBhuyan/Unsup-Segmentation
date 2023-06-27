import cv2

path = './ISIC 2016 for maskgen/IMAGES/ISIC_0000043.jpg'
img=cv2.imread(path,cv2.IMREAD_COLOR)

grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
#Black hat filter
kernel = cv2.getStructuringElement(1,(9,9)) 
blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
bhg= cv2.GaussianBlur(blackhat,(3,3),cv2.BORDER_DEFAULT)
ret,mask = cv2.threshold(bhg,10,255,cv2.THRESH_BINARY)
dst = cv2.inpaint(img,mask,6,cv2.INPAINT_TELEA)   

cv2.imshow("Original image",img)
cv2.imshow("Binary mask",mask)
cv2.imshow("Clean image",dst)

cv2.imwrite('DullRazor_'+path, dst)

cv2.waitKey()
cv2.destroyAllWindows()

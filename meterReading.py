from cv2 import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

img = cv2.imread("Resources/meter4.jpeg")

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

medianImg = cv2.medianBlur(imgGray,5)

img_blur = cv2.GaussianBlur(medianImg, (7, 7), 0)
imgCanny = cv2.Canny(img_blur,50,50)
imgConture_copy = img.copy()

#getting the cropped image
def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            cv2.drawContours(imgConture_copy, cnt, -1, (0, 0, 255), 1)
            peri = cv2.arcLength(cnt,True)
            print(area)
            print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            print(approx)
            objectCorner = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            # cv2.rectangle(imgConture_copy,(x,y),(x+w,y+h),(0,255,0),2)
            ROI = imgConture_copy[y:y + h, x:x + w]
            cv2.imwrite('Resources/ROI_{}.png'.format(1), ROI)

# def getNumbers(img):
#     img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     #hImage, wImage= img_rgb.shape
#     boxes = pytesseract.image_to_data(img_rgb)
#     print(boxes)
#     for b in boxes.splitlines():
#         print(b)
#         x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
#         cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(0,0,255),1)

    # cv2.imshow("Doing",img_rgb)

getContours(imgCanny)

cropped_Image = cv2.imread('Resources/ROI_1.png')
# getNumbers(cropped_Image)

cv2.imshow("Filtered image",imgConture_copy)
cv2.imshow("Cropped Image",cropped_Image)
cv2.waitKey(0)

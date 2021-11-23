import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

class PLFinder:
    img = None
    reader = None

    #init and preprocessing
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        pass


    def preprocess(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
        #plt.show()
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)
        plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
        #plt.show()
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            if len(approx) == 4 and approx[0][0][1] > 100 and ar>2:
                location = approx
                break
        if location is not None:
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [location], 0, 255, -1)
            new_image = cv2.bitwise_and(self.img, self.img, mask=mask)
            plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
            #plt.show()
            (x, y) = np.where(mask == 255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
            plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            #plt.show()

            self.img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    def set_image(self,image):
        self.img = image

    def next_image(self,image):
        self.set_image(image)
        self.preprocess()

    def get_image(self):
        return self.img

    def OCR(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        result = self.reader.readtext(gray)
        return result



if __name__ == "__main__":
    img = cv2.imread('seq\\frame195.jpg')
    PL = PLFinder()
    PL.next_image(img)
    print(PL.OCR())

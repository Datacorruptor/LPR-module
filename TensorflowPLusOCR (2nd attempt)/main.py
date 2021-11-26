#@title Imports and function definitions

# For running inference on the TF-Hub module.
import re

import tensorflow as tf
import pickle
import tensorflow_hub as hub
import cv2
# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time

#preprocessing and OCR
def recognize_plate(img, coords):
    # separate coordinates from box
    plate_num = ""
    try:
        xmin, ymin, xmax, ymax = coords
        print( xmin, ymin, xmax, ymax)
        print(type(img))
        print(img)
        # get the subimage that makes up the bounded region
        box = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        # grayscale region within bounding box
        gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
        # resize image to three times as large as original for better readability
        gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
        # perform gaussian blur to smoothen image
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        #cv2.imshow("Gray", gray)
        #cv2.waitKey(0)
        # threshold the image using Otsus method to preprocess for tesseract
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        #cv2.imshow("Otsu Threshold", thresh)
        #cv2.waitKey(0)
        # create rectangular kernel for dilation
        rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # apply dilation to make regions more clear
        dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
        #cv2.imshow("Dilation", dilation)
        #cv2.waitKey(0)
        # find contours of regions of interest within license plate
        try:
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except:
            ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # sort contours left-to-right
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        # create copy of gray image
        im2 = gray.copy()
        # create blank string to hold license plate number

        # loop through contours and find individual letters and numbers in license plate
        for cnt in sorted_contours:
            x,y,w,h = cv2.boundingRect(cnt)
            height, width = im2.shape
            # if height of box is not tall enough relative to total height then skip
            if height / float(h) > 6: continue

            ratio = h / float(w)
            # if height to width ratio is less than 1.3 skip
            if ratio < 1.3: continue

            # if width is not wide enough relative to total width then skip
            if width / float(w) > 25: continue

            area = h * w
            # if area is less than 100 pixels skip
            if area < 100: continue

            # draw the rectangle
            rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
            # grab character region of image
            try:
                roi = thresh[y-5:y+h+5, x-5:x+w+5]
                # perfrom bitwise not to flip image to black text on white background
                roi = cv2.bitwise_not(roi)
                # perform another blur on character region
                roi = cv2.medianBlur(roi, 5)
                try:
                    text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCEHKMOPTXY --psm 8 --oem 3')
                    # clean tesseract text by removing any unwanted blank spaces
                    clean_text = re.sub('[\W_]+', '', text)
                    plate_num += clean_text
                except:
                    text = None
            except:
                text = None
        if plate_num != None and len(plate_num)>=8 and len(plate_num) <=9:
            for i in range(len(plate_num)):
                if i in [1,2,3,6,7,8]:
                    if not plate_num[i].isdigit():
                        return None
                elif plate_num[i].isdigit():
                    return None

            print("License Plate #: ", plate_num)
            cv2.imwrite("plates\\plate_"+plate_num+".jpg",box)

        #cv2.imshow("Character's Segmented", im2)
        #cv2.waitKey(0)
    except Exception as e:
        pass
    return plate_num



def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)

  PLT_NUM = recognize_plate(np.array(image).copy(), (left, top, right, bottom))

  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)



  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin
  return PLT_NUM


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  font = ImageFont.load_default()
  PLT_NUM = None
  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      if class_names[i].decode("ascii")  != "Vehicle registration plate":
        continue
      display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                     int(100 * scores[i]))
      color = "#f4a460"
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      PLT_NUM = draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
      break
  return image , PLT_NUM

print("loading")
x = hub.load("resnet")
print("middle")
detector = x.signatures['default']
print("loaded")

def run_detector(detector, img):
  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  start_time = time.time()
  result = detector(converted_img)
  end_time = time.time()
  result = {key:value.numpy() for key,value in result.items()}
  print("Found %d objects." % len(result["detection_scores"]))
  print("Inference time: ", end_time-start_time)


  image_with_boxes , plate = draw_boxes(img, result["detection_boxes"],result["detection_class_entities"], result["detection_scores"])

  return image_with_boxes , plate


def detect_img(image):
  start_time = time.time()
  image_with_boxes , plate = run_detector(detector, image)
  end_time = time.time()
  print("Inference time:",end_time-start_time)
  return image_with_boxes, plate


plates = dict()
vidcap = cv2.VideoCapture(r"files\Untitled.MP4")


success, image = vidcap.read()
count = 0
while success:

    success, image = vidcap.read()
    if not success:
        break

    frame,plate = detect_img(image)
    cv2.imshow("Frame", image)

    print(plate)
    if plate not in plates:
        plates[plate] = 1
    else:
        plates[plate] += 1
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    print('count:', count)
    count += 1

print(plates)

detect_img(cv2.imread(r"files\seq\frame236.jpg"))

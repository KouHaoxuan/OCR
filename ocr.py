import cv2
import matplotlib.pyplot as plt
import numpy as np
from paddleocr import PaddleOCR, draw_ocr


# Functions Definition
# Convert the image to grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Use median bluring to remove noise from the image
def remove_noise(image):
    return cv2.medianBlur(image,5)

#OTSU thresholding
def thresholding(image):
    return cv2.threshold(image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
"""
Usually the processed image is color-balanced, and it is more appropriate to set the threshold to 127 directly.
However, sometimes the distribution of gray levels of the image is not balanced, if the threshold is also set to 127, then the result of thresholding is a failure.
The OTSU method iterates over all possible thresholds to find the optimal one. When using the OTSU method, set the threshold to 0. The function cv2.threshold() at this point will automatically find the optimal threshold and return that threshold.
"""

# Opening operation means erosion followed by dilation
# Opening operation can smooth contours, break narrow breaks and eliminate small protrusions, remove off-target isolation points
def opening(image):
    kernel = np.ones((5, 5), np.uint8)#initialize the kernal
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Closing is the opposite of opening operation. It means dilation follow by erosion.
# It can fill small cavities within objects, connect adjacent objects, smooth their boundaries and remove internal sharp corners without significantly altering their area
def closing(image):
    kernel = np.ones((2, 2), np.uint8)#initialize the kernal
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

#Canny edge dection
def canny(image):
    return cv2.Canny(image, 100, 200)

#Robert edge dection
def robert(image):
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)# Defining Robert operator
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(image, cv2.CV_16S, kernelx)
    y = cv2.filter2D(image, cv2.CV_16S, kernely)
    # Convert to uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Roberts

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))# Find all non-zero pixel coordinates in the image
    angle = cv2.minAreaRect(coords)[-1]# Calculate the minimum area bounding rectangle for the coordinates. The function returns (center (cx, cy), (width, height), angle of rotation)
    if angle < -45:# Correct the angle if it's more than -45 degrees (to ensure it rotates clockwise)
        angle = -(90 + angle)
    else:
        angle = -angle# Make it positive for clockwise rotation
    (h, w) = image.shape[:2] # Get the dimensions of the image
    center = (w // 2, h // 2)# Calculate the center of the image
    M = cv2.getRotationMatrix2D(center, angle, 1.0) # Create a rotation matrix for the given center, angle, and scale (1.0 for no scaling)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)# Apply the affine transformation to rotate the image.Use cubic interpolation and replicate the border pixels
    return rotated







#image pre-processing
image = cv2.imread('image.jpg')
gray = get_grayscale(image)#convert the image to grayscale image
thresh = thresholding(gray)#OTSU Threshold
opening = opening(gray)#opening operation to the image
closing = closing(gray)#closing operation to the image
canny = canny(opening)#Canny edge detection
robert = robert(opening)#Robert edge detection
#deskew = deskew(image)

cv2.imshow('image',image)
cv2.waitKey(0)

#show and save the image
cv2.imshow('gray',gray)
cv2.waitKey(0)
cv2.imwrite('gray.jpg',gray)

#show and save the image
cv2.imshow('open',opening)
cv2.waitKey(0)
cv2.imwrite('open.jpg',opening)

#show and save the image
cv2.imshow('close',closing)
cv2.waitKey(0)
cv2.imwrite('close.jpg',closing)

#show and save the image
cv2.imshow('canny',canny)
cv2.waitKey(0)
cv2.imwrite('canny.jpg',canny)

#show and save the image
cv2.imshow('robert',robert)
cv2.waitKey(0)
cv2.imwrite('robert.jpg',robert)









# Text Recognition
# Paddleocr currently supports multiple languages that can be switched by modifying the lang parameter.
# For example, ‘ch’, ‘en’, ‘fr’, ‘german’, ‘korean’, ‘japan’.
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False,
                lang="ch")  # need to run only once to download and load model into memory

# Text Recognition using original image
img_path1 = 'image.jpg'
result = ocr.ocr(img_path1, cls=True)
for i in result:
    for line in i:
        print(line)

# Text Recognition using grayscale image
img_path2 = 'gray.jpg'
result = ocr.ocr(img_path2, cls=True)
for i in result:
    for line in i:
        print(line)

# Text Recognition using image after opening operation
img_path3 = 'open.jpg'
result = ocr.ocr(img_path3, cls=True)
for i in result:
    for line in i:
        print(line)






result = ocr.ocr(img_path2, cls=True)
# Save the text in a txt file
# Initialize text content
output_text = ""

# Iterate through the recognition results
for line in result[0]:  # result[0] contains all detected lines of text in the image
        bbox, text = line[0], line[1][0]  # bbox the bound boxes，text is the recognized text
        output_text += f"boxes: {bbox}\ntext: {text}\n\n"
with open('output.txt', 'w', encoding='utf-8') as file:
    file.write(output_text)
print("Recognition results are written to output.txt")









# Template Matching
img_path = 'gray.jpg'
img = cv2.imread(img_path)

# Perform text detection and recognition using PaddleOCR.
# The 'cls=True' parameter enables text direction classification.
result = ocr.ocr(img_path, cls=True)

# Extract text content
texts = [line[1][0] for line in result[0]]  # Suppose we only care about the content of the first line of text

# text we need to match
target_text = '人生'

# Simple String Matching
for text in texts:
    if target_text in text:  # Partial matches are used here, you can change it to full matches etc. as needed
        print(f'Match to text: {text}')
        break
else:
    print('No text matched')









# Using the draw_ocr function to visualize OCR results
from PIL import Image

boxes = [line[0] for line in result[0]]# Extract bounding boxes for detected text lines.
txts = [line[1][0] for line in result[0]]# Extract recognized text for each detected line.
scores = [line[1][1] for line in result[0]]# Extract confidence scores for each recognized text.

# Optionally draw the detected text boxes on the image.
# You can specify a font path for drawing if needed.
im_show = draw_ocr(img, boxes)
cv2.imshow('OCR Result', im_show)# Display the image with OCR results.
cv2.waitKey(0)# Wait for a key press indefinitely.
cv2.destroyAllWindows()





image = Image.open(img_path).convert('RGB')
boxes = [detection[0] for line in result for detection in line]  # Nested loop added
txts = [detection[1][0] for line in result for detection in line]  # Nested loop added
scores = [detection[1][1] for line in result for detection in line]  # Nested loop added
im_show = draw_ocr(image, boxes, txts, scores)
im_show = Image.fromarray(im_show)
im_show.save('test.jpg')
plt.imshow(im_show)
plt.axis('off')
plt.show()




















"""
# Paddleocr currently supports multiple languages that can be switched by modifying the lang parameter.
# For example, ‘ch’, ‘en’, ‘fr’, ‘german’, ‘korean’, ‘japan’.
PAGE_NUM = 10 #  Set the recognition page number
pdf_path = 'cv.pdf'
ocr = PaddleOCR(use_angle_cls=True,use_gpu=False, lang="en", page_num=PAGE_NUM)  # need to run only once to download and load model into memory
# ocr = PaddleOCR(use_angle_cls=True, lang="ch", page_num=PAGE_NUM,use_gpu=0) # To Use GPU,uncomment this line and comment the above one.
result = ocr.ocr(pdf_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    if res == None: # Skip when empty result detected to avoid TypeError:NoneType
        print(f"[DEBUG] Empty page {idx+1} detected, skip it.")
        continue
    for line in res:
        print(line)


# display the result
import fitz
from PIL import Image
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr


imgs = []
with fitz.open(pdf_path) as pdf:
    # Get page number
    PAGE_NUM = pdf.page_count
    for pg in range(PAGE_NUM):
        page = pdf.load_page(pg)  # load pages
        # Setting the scaling matrix
        mat = fitz.Matrix(2, 2)  # For example, zoom in twice
        # Get the page's Pixmap object
        pm = page.get_pixmap(matrix=mat, alpha=False)
        # Check dimensions
        if pm.width > 2000 or pm.height > 2000:
            pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        # Converting a Pixmap to a PIL Image Object
        img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
        # Converting to a NumPy array and changing the color space
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_gray = get_grayscale(img_bgr)
        cv2.imshow('Page', img_gray)
        cv2.waitKey(0)
        # Add processed images to the list
        imgs.append(img_bgr)


for idx in range(len(result)):
    res = result[idx]
    if res == None:
        continue
    image = imgs[idx]
    boxes = [line[0] for line in res]
    txts = [line[1][0] for line in res]
    scores = [line[1][1] for line in res]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='doc/fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('result_page_{}.jpg'.format(idx))
    plt.imshow(im_show)
    plt.axis('off')
    plt.show()
cv2.destroyAllWindows()"""


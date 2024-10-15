import fitz
from PIL import Image
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt

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
        cv2.imshow('Page', img_bgr)
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
cv2.destroyAllWindows()
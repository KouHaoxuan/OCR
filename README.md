# OCR
Assignment1 (Base on PaddleOCR) 
1. Introductions:

1.1 Introduction of OCR:
OCR, i.e. Optical Character Recognition, is a technology capable of converting textual information in image files into electronic text.
The basic principle of OCR technology is to check the characters printed on paper through electronic devices such as scanners or digital cameras, to capture images of the characters using optical technology, and to convert these images into computer-readable text information through a series of complex algorithms.
This process usually includes steps such as image preprocessing, text detection, feature extraction, text recognition and post-processing

1.2 Introduction of Paddleocr:
In this assignment, I chose the PaddleOCR which is developed by Baidu to finish the tasks and it has a good performance. 
PaddleOCR is a powerful and open-source OCR (Optical Character Recognition) tool developed by Baidu, based on its PaddlePaddle deep learning framework.
PP-OCR is a practical and ultra-lightweight OCR system developed by PaddleOCR itself. Based on the implementation of cutting-edge algorithms, it considers the balance between accuracy and speed, and carries out model thinning and deep optimization, so that it can meet the needs of industrial landing as much as possible. The system contains two stages of text detection and text recognition, in which DB is selected for text detection algorithm and CRNN is selected for text recognition algorithm, and a text direction classifier is added between the detection and recognition modules to cope with text recognition in different directions. In this program, I used the module PP-OCRv3, and on the basis of PP-OCRv2, a total of 9 aspects have been upgraded for the detection model and recognition model to further improve the model effect.
From the effect, the speed is comparable to the case, a variety of scene accuracy has been greatly improved:
In Chinese scenes, compared to the PP-OCRv2 Chinese model improved by more than 5%;
In English digital scene, compared with the PP-OCRv2 English model to improve 11%;
In Multi-language scenarios, optimizing the recognition effect of 80+ languages, and improving the average accuracy by more than 5%.

1.3 Introduction of this Assignment:
In this assignment, the program I developed is based on PaddleOCR. It can automatically preprocess the image we input, extract text from preprocessed image and store the text content in a local txt file. This program also contains some result visualization functionalities to display how the detection process works.

The ocr.py contains functionalities of preprocessing image, extracting text from single image and storing the extracted text to a local txt file. The pdf_OCR.py allow users to extract text from pdf files. Both of them have a good performance.

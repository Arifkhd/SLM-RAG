import pytesseract
from PIL import Image
import cv2
import numpy as np

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return pytesseract.image_to_string(img)

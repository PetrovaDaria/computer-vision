import cv2
import numpy as np
from random import randrange
import time
img = np.zeros((1920, 1080), dtype = np.uint8)
counter = 0
start = time.time()
while counter < 1000:
    cv2.line(img, (randrange(0, 1920), randrange(0, 1080)), (randrange(0, 1920), randrange(0, 1080)), (randrange(0, 255)))
    cv2.imshow('test', img)
    temp = cv2.waitKey(1)
    counter += 1

print(time.time() - start)

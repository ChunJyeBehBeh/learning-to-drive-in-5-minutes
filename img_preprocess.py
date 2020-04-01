import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
from config import Grey_Only

'''
road = 50   
track= 40   (desert)
'''
ROI_y = 30


x=0

def detect_edges(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

    
def preprocess_image(image):
    global x
    grey_only = Grey_Only

    global x
    grey_only = not Edges_Detection

    if grey_only:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grey = np.stack((grey,)*3, axis=-1)
        # print("Processed Image shape: {}".format(grey.shape))
        return grey

    img=cv2.GaussianBlur(image, (5, 5), 0)

    # Simulator
    edges = detect_edges(img, low_threshold=80, high_threshold= 200)
    # edges = detect_edges(img, low_threshold=10, high_threshold= 50)
    
    # cropped edges
    edges[0:ROI_y,:]= 0         

    # edges = edges.reshape((80,160,1))
    edges = cv2.merge((edges,edges,edges))

    # cv2.imwrite("TEST/{}.jpg".format(x),edges)
    # x+=1

    return edges

 
if __name__ == '__main__':
    show = False
    preprocess = True
    img2video = False
    x = 0

    if preprocess:
        for file in os.listdir("path-to-record/level0"):
            print(file)
            image = cv2.imread("path-to-record/level3/{}".format(file))
            img=preprocess_image(image)
            cv2.imwrite("TEST/{}.jpg".format(x),img)
            x+=1
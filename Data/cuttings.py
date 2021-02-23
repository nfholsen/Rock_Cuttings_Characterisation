import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt 

from skimage import img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.measure import find_contours,regionprops, label
from skimage.morphology import disk, erosion, dilation, closing

# Class : 

class Cuttings:
    """
    
    """
    def __init__(self,root_path,image_path):
        self.root_path = root_path
        self.image_path = image_path
        
    def load_picture(self):
        image = cv2.imread(self.root_path + self.image_path, cv2.IMREAD_GRAYSCALE)
        return image
    
    def assign_mask(self,image):
        mask = image > threshold_otsu(image)
        return mask
    
    def assign_label(self,mask):
        """
        Take as input a binary image abd return image with labels
        """
        selem = disk(3)

        eroded = mask.copy()

        for i in range(0,3):
            eroded = erosion(eroded, selem)
        dilated = label(eroded).copy() 
        for i in range(0,3):
            dilated = dilation(dilated, selem)
        for i in range(0,3):    
            dilated = closing(dilated, disk(1))
        return dilated
    
    def big_cuttings(self,dilated,threshold=1000):
        """
        take as input an image with labels, return the fragments greater than 1000px 
        """
        big_samples = []
        for i in range(len(regionprops(dilated))):
            if regionprops(dilated)[i].area > threshold : big_samples.append(i)
        return big_samples


# Functions :

def crop_rectangle(im_rect,box,rect):
    W = rect[1][0]
    H = rect[1][1]
    
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    
    rotated = False
    angle = rect[2]
    
    if angle < -45:
        angle+=90
        rotated = True
        
    if angle > 45:
        angle-=90
        rotated = True

    size = (x2-x1,y2-y1)
    center = (int((x1+x2)/2), int((y1+y2)/2))
    
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    
    cropped = cv2.getRectSubPix(im_rect, size, center)    
    cropped = cv2.warpAffine(cropped, M, size)
    
    croppedW = W if not rotated else H 
    croppedH = H if not rotated else W
    
    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW), int(croppedH)), (size[0]/2, size[1]/2))
    return croppedRotated

def pad_image(im_rect_not_pad,desired_size=128):
    old_size = im_rect_not_pad.shape[:2]
    
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im_rect_not_pad = cv2.resize(im_rect_not_pad, (new_size[1], new_size[0]),interpolation=cv2.INTER_CUBIC)
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = 0
    
    im_rect_pad = cv2.copyMakeBorder(im_rect_not_pad, top, bottom, left, right, cv2.BORDER_CONSTANT,
    value=color)
    return im_rect_pad

def plot_samples(image,label_image,im):
    minr, minc, maxr, maxc = regionprops(label_image)[im].bbox
        
    fig,ax = plt.subplots(1,2)
        
    ax[0].imshow(image[minr:maxr,minc:maxc]*regionprops(label_image)[im].image,
                cmap=plt.cm.gray,aspect="auto")
    ax[1].imshow(regionprops(label_image)[im].image,
                cmap=plt.cm.gray,aspect="auto")
    plt.show()

def plot_samples_resized(image,label_image,im,WIDTH=128,HEIGHT=128): 
    minr, minc, maxr, maxc = regionprops(label_image)[im].bbox
        
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(cv2.resize(image[minr:maxr,minc:maxc]*regionprops(label_image)[im].image,
                   (WIDTH,HEIGHT), 
                   interpolation=cv2.INTER_CUBIC),
                cmap='gray')
    ax[1].imshow(cv2.resize(regionprops(label_image)[im].image.astype(np.uint8),
                   (WIDTH,HEIGHT), 
                   interpolation=cv2.INTER_CUBIC),
                cmap='gray')
    plt.show()

def plot_samples_padded(image,label_image,im):
    minr, minc, maxr, maxc = regionprops(label_image)[im].bbox
        
    fig,ax = plt.subplots(1,2)
        
    ax[0].imshow(pad_image(image[minr:maxr,minc:maxc]*regionprops(label_image)[im].image),
                cmap=plt.cm.gray,aspect="auto")
    ax[1].imshow(pad_image(regionprops(label_image)[im].image.astype(np.uint8)),
                cmap=plt.cm.gray,aspect="auto")
    plt.show()
    
def plot_rectangular_samples(image,dilated,im):
    im_rect = img_as_ubyte(dilated == regionprops(dilated)[im].label)#.astype(np.uint8)
    contours,_ = cv2.findContours(im_rect, 1, 2)
    rect = cv2.minAreaRect(contours[0])
    if len(contours) >= 2:
        cmax = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        rect = cv2.minAreaRect(cmax)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
       
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(crop_rectangle(im_rect,box,rect),
                cmap='gray')
    ax[1].imshow(crop_rectangle(image*im_rect,box,rect),
                cmap='gray')
    plt.show()
    
def plot_rectangular_samples_resized(image,dilated,im,WIDTH = 128,HEIGHT = 128):
    im_rect = img_as_ubyte(dilated == regionprops(dilated)[im].label)#.astype(np.uint8)
    contours,_ = cv2.findContours(im_rect, 1, 2)
    rect = cv2.minAreaRect(contours[0])
    if len(contours) >= 2:
        cmax = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        rect = cv2.minAreaRect(cmax)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
        
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(cv2.resize(crop_rectangle(im_rect,box,rect),
                   (WIDTH,HEIGHT), 
                   interpolation=cv2.INTER_CUBIC),
                cmap='gray')
    ax[1].imshow(cv2.resize(crop_rectangle(image*im_rect,box,rect),
                   (WIDTH,HEIGHT), 
                   interpolation=cv2.INTER_CUBIC),
                cmap='gray')
    plt.show()
    
def plot_rectangular_samples_padded(image,dilated,im):
    im_rect = img_as_ubyte(dilated == regionprops(dilated)[im].label)#.astype(np.uint8)
    contours,_ = cv2.findContours(im_rect, 1, 2)
    rect = cv2.minAreaRect(contours[0])
    if len(contours) >= 2:
        cmax = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        rect = cv2.minAreaRect(cmax)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
       
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(pad_image(crop_rectangle(im_rect,box,rect)),
                cmap='gray')
    ax[1].imshow(pad_image(crop_rectangle(image*im_rect,box,rect)),
                cmap='gray')
    plt.show()
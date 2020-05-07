import cv2
import numpy as np
import os
from os.path import isfile, join
from tqdm import tqdm
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ 
    Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ 
    Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

if __name__ == '__main__':

    pathIn= './predictions/'
    pathOut = 'video.avi'
    fps = 60

    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    #for sorting the file names properly
    sort_nicely(files)

    for i in tqdm(files):
        filename = pathIn + i

        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])

    out.release()
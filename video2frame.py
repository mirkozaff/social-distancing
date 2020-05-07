import cv2
import os.path as path

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        # save frame as JPG file
        cv2.imwrite(f'video_frames/image_{count}.jpg', image)     
    return hasFrames

if __name__ == '__main__':
    #video to extract
    vidcap = cv2.VideoCapture('shopping.mp4')

    sec = 0
    frameRate = 1/60 # it will capture 30 frames per second
    count = 1
    success = getFrame(sec)
    while success and sec < 30:
        count = count + 1
        sec += frameRate
        sec = round(sec, 2)
        success = getFrame(sec)
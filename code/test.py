import pandas as pd
from time import sleep
from os import SEEK_END, SEEK_CUR
import pytesseract
import cv2 as cv
import draw_bounding_box
from pathlib import Path
import numpy as np
import mss
from os import getcwd
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

cv.destroyAllWindows()

debug = True
level_counter = 0

def countup(next_level_count):
    print('counmt up')
    filename = 'GPB_1LifeChallenge'
    date = datetime.now().strftime('%Y-%m-%d')# This will help to make a specific file for a specific game to promote abstraction in the code
    myfile = Path( filename + '.csv') # assigns the variable myfile
    myfile.touch(exist_ok=True) # If it already exists, don't do anything, else create it.
    
    print(next_level_count)
    
    next_level_count = next_level_count.replace('\n','')
    next_level_count = int(next_level_count)
    
    data = {
        'date':[date],
        'timestamp_of_video':[np.nan],
        'level_count':[next_level_count],
        'run_number':[np.nan],
        'auto_comment':['place_holder']
    }
    
    try:
        df = pd.read_csv(myfile) 
        new_old_file = 'old'
    except pd.errors.EmptyDataError:
        data['date'] = [date]
        data['timestamp_of_video'] = [np.nan]
        data['level_count'] = [next_level_count]
        data['run_number'] = 1
        data['auto_comment'] = 'number detected but did not meet criteria of new run or next level'
        df = pd.DataFrame(data)
        new_old_file = 'new'
        print('new file')
        
    last_row = df.tail(1)
    if next_level_count == last_row['level_count'].any() + 1:
        data['date'] = [date]
        data['timestamp_of_video'] = [np.nan]
        data['level_count'] = [next_level_count]
        data['run_number'] = last_row['run_number']
        data['auto_comment'] = 'Next Level'
    elif ((next_level_count == 0) and (last_row['level_count'].any() != 0)):
        data['date'] = [date]
        data['timestamp_of_video'] = [np.nan]
        data['level_count'] = [next_level_count]
        if new_old_file == 'new':
            data['run_number'] = 1
        else:
            data['run_number'] = last_row['run_number']+1
        data['auto_comment'] = 'Run Ended'
    
    
    df = pd.DataFrame(data)
    
    df.to_csv(myfile, mode='a',)
    
    return


    myfile = Path( date + '.txt') # assigns the variable myfile to gamefilename.txt


    myfile.touch(exist_ok=True) # If it already exists, don't do anything, else create it.
    with open(myfile, 'rb') as f:
        try: # Cathc OS Error in case of a one line file
            f.seek(-2, SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
    print(last_line)
        
    last_line_dict = {
        'date':last_line.split(',')[0],
        'timestamp_of_video':last_line.split(',')[1],
        'level_count':int(last_line.split(',')[2]),
        'run_number':int(last_line.split(',')[3]),
        'auto_comment':last_line.split(',')[3]
    }
        


        

def bgremoval(myimage):
    
    
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    image_in_hsv = cv.cvtColor(myimage, cv.COLOR_BGR2HSV)
    mask = cv.inRange(image_in_hsv, low_green, high_green)
    mask = 255-mask
    # res = cv2.bitwise_and(img, img, mask=mask)
 
    return mask
    

def main():
    text_gray = -1
    sct = mss.mss()
    if debug==True:
        stream_coords = [(50, 161), (1247, 835)]
        score_coords = [(240+55, 199), (564, 253)]
    else:
        stream_coords = draw_bounding_box.main()
        sleep(.2)
        score_coords = draw_bounding_box.main()
    
    
    stream_dimensions = {
        'left':stream_coords[0][0],
        'top':stream_coords[0][1],
        'right':stream_coords[1][0],
        'bottom':stream_coords[1][1],
        'width':(stream_coords[1][0]-stream_coords[0][0]),
        'height':(stream_coords[1][1]-stream_coords[0][1]),
    }
    
    score_dimensions = {
        'left':score_coords[0][0],
        'top':score_coords[0][1],
        'right':score_coords[1][0],
        'bottom':score_coords[1][1],
        'width':(score_coords[1][0]-score_coords[0][0]),
        'height':(score_coords[1][1]-score_coords[0][1]),
    }
    
    
    
    

    

    while True:
        stream_gray = np.array(sct.grab(stream_dimensions))
        cv.imshow('stream', cv.cvtColor(stream_gray, cv.COLOR_BGR2GRAY))
        score_orig = np.array(sct.grab(score_dimensions))
        score_orig = cv.medianBlur(score_orig, 5)
        
        
        score_orig_no_background = bgremoval(score_orig)
        
        
        
        cv.imshow('removebackground', score_orig_no_background)
        
        
        
        
        score_gray = np.array(sct.grab(score_dimensions))
        score_gray = cv.cvtColor(score_gray, cv.COLOR_BGR2GRAY)
        score_gray = cv.GaussianBlur(score_gray, (5,5), 0)
        # score_gray = cv.Canny(score_gray, 100, 10)
        # score_gray = cv.medianBlur(score_gray, 7)
        # score_gray = cv.Canny(score_gray, 100,300)
        
        text_gray = pytesseract.image_to_string(score_orig_no_background,  config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789')
        # print(type(text_gray))
        if text_gray == '':
            pass
        if text_gray != '':
            int(text_gray)
            countup(text_gray)
        
        # if text_gray == None:
        #     print('None as type')
        # if text_gray == "None":
        #     print('None as string')
        # print(b)
        # text_gray = int(text_gray)
        
        # if text_gray > text_gray or text_gray == 0:
            # print(text_gray)
            
        cv.imshow('Computer Vision', score_orig)



            
        if cv.waitKey(1) == ord("q"):
            break
        
        
if __name__ == "__main__":
    main()
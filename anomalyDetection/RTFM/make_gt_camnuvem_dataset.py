import os
import numpy as np
import cv2
import re
from collections import Counter


"""
The Camnuvem dataset has only one category
For each i3d vector in test folder, open the labels txt file, read the relative position of each abnormality,
and create a binary vector for each frame in video. 


"""


gt = []

# Given a video path, count the frames
#  THE cv2.CAP_PROP_FRAME_COUNT may give a different frame qtd than the real readed qtd (what can result some problems
# when the last frame is sampled in the feature extraction phase). Thus, we have to read the frame qtd 
# iterating frame per frame.
def countFrame(video_path):

    # Opens the Video file
    cap= cv2.VideoCapture(video_path)
    frameQtd = 0

    while(cap.isOpened()):

        ret, frame = cap.read()
        if ret == 0:
            break
        frameQtd += 1
     
    cap.release()
    cv2.destroyAllWindows() 

    return frameQtd



# Carrega as posições das anomalias nos vídeos
abnormalities = []
video_path = []
def readAbnormalities(test_labels, video_root_test_abnormal):

    with open(test_labels) as file:
        lines = file.readlines()

    for video in lines:
        line_vec = video.split()
        abnormalities.append(line_vec[1:])    # Abnormalities in terms of porcentage
        video_path.append(os.path.join(video_root_test_abnormal, line_vec[0]))

def findMp4Path(npyPath, isnormal, video_root_test_normal, video_root_test_abnormal):

    print(npyPath)

    if isnormal:
        result =  re.search('normal/(.*).npy', npyPath)
        result = result.group(1)        
        return video_root_test_normal + "/" + result + ".mp4"
    else:
        result =  re.search('anomaly/(.*).npy', npyPath)
        result = result.group(1)                
        return video_root_test_abnormal + "/" + result + ".mp4"


def findAbnormalitie(npyPath):
    result =  re.search('anomaly/(.*).npy', npyPath)
    result = result.group(1)

    for i in range(len(video_path)):
        path = video_path[i]

        if str(result)+".mp4"  in path:
            return i


    return -1

# Return True if the video is normal, false otherwise
def verifyNormal(video_path):
    return "normal" in video_path

cont = 0

def start(NUM_FRAMES_IN_EACH_FEATURE_VECTOR, list_test, test_labels, video_root_test_abnormal, video_root_test_normal, output_file):
    global gt
    with open(list_test) as list_test_files:
        npy_files = list_test_files.readlines()

    readAbnormalities(test_labels, video_root_test_abnormal)
    print(abnormalities)
    print(video_path)

    # If everything is going alright, lines has the same lengh than videos in video_root_test_abnormal
    for npy in npy_files:
        npy = npy.rstrip("\n")
        print(npy)


        #line_vec = video.split()
        #abnormalities = line_vec[1:]    # Abnormalities in terms of porcentage
        #video_path = os.path.join(video_root_test_abnormal, line_vec[0])


        # Read npy
        #feature_name = os.path.join(i3d_root_path_anomaly, line_vec[0][:-4]+".npy")
        # Feature has (segments, 10, 2048). Each segment has NUM_FRAMES_IN_EACH_FEATURE_VECTOR frames.
        features = np.load(npy)

        # Create the ground truth
        # 1. First lets create a vector with same number of elements than frames in video
        # 2. Then lets iterate all abnormalitie in line_vec and replace only the right positions to 1
        
        # Step 1.
        gt_ = np.zeros(NUM_FRAMES_IN_EACH_FEATURE_VECTOR*features.shape[0]).astype(float)

        video_path_ = findMp4Path(npy, verifyNormal(npy), video_root_test_normal, video_root_test_abnormal)    
        frame_qtd = countFrame(video_path_)

        if verifyNormal(npy) is False:

            # Find the abnormalities of 'video' in 'abnormalities'
            index = findAbnormalitie(npy)
            assert(index > -1)
            #video_path_ = video_path[index]
            abnormalities_ = abnormalities[index]

            # We need to know the total frames in each video. So, we have to open each video
            

            # Step 2.
            for i in range(int(len(abnormalities_)/2)):    # Each abnormalitie has the init and end position, this the reason of the division by 2

                init = float(abnormalities_[i*2])
                end = float(abnormalities_[(i*2) + 1])
                frame_init = round(frame_qtd * init) -1     # Frames starts at 0
                frame_end = round(frame_qtd * end) -1       # Frames starts at 0

                gt_[frame_init:frame_end] = 1.0
                print(video_path_)
                print(frame_qtd)
                print(init)
                print(end)
                print(frame_init)
                print(frame_end)
                print("ok, ncontramos um 1 na posicao " + str(frame_init))

                print(gt_)

        oversampled = frame_qtd % NUM_FRAMES_IN_EACH_FEATURE_VECTOR 
        #print("Oversampled: " + str(oversampled))
        #print("Dimensão do gt_: " + str(len(gt_)))
        #print("index inicial do gt_: " + str(len(gt_)-oversampled-1))
        if oversampled > 0:
            # copy the last labelled frame to the other sampled frames
            referenceLabelIndex = len(gt_)-oversampled - 1
            gt_[referenceLabelIndex + 1 : ] = [gt_[referenceLabelIndex] for i in range(oversampled)]

        gt.extend(gt_)

        #exit()

    print("PORRRRRRRRRRRRA")
    print(Counter(gt))

    #output_file = '/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/RTFM/list/gt-camnuvem.npy'
    gt = np.array(gt, dtype=float)
    np.save(output_file, gt)
    print(len(gt))

if __name__ == '__main__':

    NUM_FRAMES_IN_EACH_FEATURE_VECTOR = 16
    list_test = "/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/RTFM/list/camnuvem-i3d-test-10crop.list"

    test_labels = "/media/denis/526E10CC6E10AAAD/CamNuvem/violent_thefts_dataset/videos/labels/test.txt"
    video_root_test_abnormal = "/media/denis/526E10CC6E10AAAD/CamNuvem/violent_thefts_dataset/videos/samples/test/anomaly"
    video_root_test_normal = "/media/denis/526E10CC6E10AAAD/CamNuvem/violent_thefts_dataset/videos/samples/test/normal"

    output_file = '/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/RTFM/list/gt-camnuvem.npy'
    start(NUM_FRAMES_IN_EACH_FEATURE_VECTOR, list_test, test_labels, video_root_test_abnormal, video_root_test_normal, output_file)
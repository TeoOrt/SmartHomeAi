# %%
import cv2
import numpy as np
import os
from training_classifier import Framer
from label_mapper import Label_mapper, Training_map
from itertools import repeat
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from classifier import Classifier

# from handshape_feature_extractor import HandShapeFeatureExtractor as FeatExt
from frameextractor import frameExtractor,frameExtractorWithInvert
import time
## import the handfeature extractor class
from handshape_feature_extractor import HandShapeFeatureExtractor as FeatExt
from concurrent.futures import ThreadPoolExecutor


from hfe import HandShapeFeatureExtractor as HFe2


# %%
# lets map training_sets with our datasets

training_map = Training_map().get_map()

# %%
# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

# lets get the video features and load in video

# lets get the videos

if not os.path.exists('train_data'):
    os.mkdir('train_data')
path = os.getcwd() + '/train_data'
dir_list = os.listdir(path)

video_labels:dict = {}

# add the counter to the videos
for videos in dir_list:
    video_labels[videos] = int(videos.split('-')[0][1]) #this is to get the counter numbers


training_path = 'training_frames'
if not os.path.exists(training_path):
    os.mkdir(training_path)
frame_getter = Framer()

for key,value in video_labels.items():
    frame_getter.get_frames(key,value) 


predictionMap = frame_getter.get_label_dump()
counter = 0 
for values in predictionMap.values():
    print(type(values))
    counter += len(values)

print(counter)

    # writer.writelines(predictionMap.items())
exit()
# get images in parallel

print(f'Recorded --> {len(training_frames)}frames for our training data')

# %%

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video
if not os.path.exists('test'):
    os.mkdir('test')

testing_path = os.getcwd() + '/test'
dir_list = os.listdir(testing_path)
test_path = 'test_frames'
if not os.path.exists('test_frames'):
    os.mkdir(test_path)

frame_getter.set_save_path(testing_path,test_path)
test_frames = []

for counter,videos in enumerate(dir_list):
    result = frame_getter.get_frames(videos,counter)
    test_frames.append(result)

# %%
# prediction = FeatExt().get_instance()
pred = HFe2().get_instance()

training_sample  =[ pred.get_instance().extract_feature(x) for x in training_frames]
testing_sample = [pred.get_instance().extract_feature(x) for x in test_frames]
exit()

# %%

# %%
# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

classifier = Classifier(training_set,testing_set)

results_array = []

for tests in testing_set.keys():
    results_array.append((tests,classifier.match_closest_result(tests)))




# %%
result_mapper = Label_mapper()


import pandas as pd
result_list = []
for values in results_array:
    for tester in values[1::2]:
        for vals in tester:
            if len(result_list) == 51:
                break
            resser = result_mapper[vals[0]]
            result_list.append(resser)



# --------------------------------------------------------
#               Write to csv file predicitions             
# --------------------------------------------------------

df = pd.DataFrame(result_list)
df.to_csv('Results.csv',index=False,header=False)



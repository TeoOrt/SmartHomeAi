# %%
import cv2
import numpy as np
import os
import tensorflow as tf
from itertools import repeat
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# from handshape_feature_extractor import HandShapeFeatureExtractor as FeatExt
from frameextractor import frameExtractor,frameExtractorWithInvert
import time
## import the handfeature extractor class
from handshape_feature_extractor import HandShapeFeatureExtractor as FeatExt
from concurrent.futures import ThreadPoolExecutor


from hfe import HandShapeFeatureExtractor as HFe2


# %%
# lets map training_sets with our datasets

training_map = dict()
training_map['DecreaseFanSpeed'] = 'H-DecreaseFanSpeed'
training_map['IncreaseFanSpeed'] = 'H-IncreaseFanSpeed'
training_map['Temperature'] = 'H-SetThermo'
training_map['FanOn'] = 'H-FanOn'
training_map['FanOff'] = 'H-FanOff'
training_map['LightOn'] = 'H-LightOn'
training_map['LightOff'] = 'H-LightOff'
training_map['Zerotimes'] = 'H-0'
training_map['Onetimes'] = 'H-1'
training_map['Twotimes'] = 'H-2'
training_map['Threetimes'] = 'H-3'
training_map['Fourtimes'] = 'H-4'
training_map['Fivetimes'] ='H-5'
training_map['Sixtimes'] = 'H-6'
training_map['Seventimes'] ='H-7'
training_map['Eighttimes'] = 'H-8'
training_map['Ninetimes'] = 'H-9'

reverse_mapper = dict()

for keys,values in training_map.items():
    reverse_mapper[values]=keys

# %%
# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

# lets get the video features and load in video

# lets get the videos
path = os.getcwd() + '/train_data'
dir_list = os.listdir(path)

video_labels:dict = {}

# add the counter to the videos
for videos in dir_list:
    
    video_labels[videos] = int(videos.split('_')[-2])

training_path = 'training_frames'
if not os.path.exists(training_path):
    os.mkdir(training_path)


def get_frames(args):
    video,counter,path = args
    video_path = path+'/'+video
    mapper_match = video.split('_')[1:-3]
    map_match = "".join(mapper_match)
    if 'Light' in video or 'Fan' in video:
        vid_splitter = video.split('_')
        name = '/'+vid_splitter[1] + vid_splitter[2]+'/'
    else:
        name = '/'+video.split('_')[1]+'/'
    frame = frameExtractorWithInvert(video_path,training_path+name,counter)
    return (frame,training_map[map_match])

# get images in parallel
predictionMap = {}
# with ThreadPoolExecutor() as tp:
    # frameArray = list(tp.map(get_frames,zip(video_labels.keys(),video_labels.values(),repeat(path))))
frameArray =[]
for key,value in video_labels.items():
    res = get_frames((key,value,path))
    frameArray.append(res)

for values in frameArray:
    if values[1] in predictionMap:
        predictionMap[values[1]].append(values[0])
    else:
        predictionMap[values[1]] = [values[0]]

print('Recorded Training sets')



# %%

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video



testing_path = os.getcwd() + '/test'
dir_list = os.listdir(testing_path)

test_labels:dict = {}

# add the counter to the videos
for videos in dir_list:
    test_labels[videos] = int(videos.split('-')[0][1]) #this is to get the counter numbers


test_path = 'test_frames'
if not os.path.exists(test_path):
    os.mkdir(test_path)


testingMap = {}
def get_frames(args):
    video,counter,path = args
    video_path = testing_path+'/'+video
    name = '/'+video[3:-4]+'/'
    mapMatcher = video.split('.')[0][3::]
    frame = frameExtractor(video_path,test_path+name,counter)
    return (frame,mapMatcher)

# get images in parallel
with ThreadPoolExecutor() as tp:
    testFrameArray = list(tp.map(get_frames,zip(test_labels.keys(),test_labels.values(),repeat(test_path))))

for values in testFrameArray:
    if values[1] in testingMap:
        testingMap[values[1]].append(values[0])
    else:
        testingMap[values[1]] = [values[0]]



# %%
# prediction = FeatExt().get_instance()
pred = HFe2().get_instance()

training_set ={}
testing_set ={}


for values in predictionMap.keys():
    training_set[values] = [pred.get_instance().extract_feature(x) for x in predictionMap[values]]




# %%
for values in testingMap.keys():
    testing_set[values] = [pred.get_instance().extract_feature(x) for x in testingMap[values]]

# %%
class Classifier():
    def __init__(self,training_map,testing_map) -> None:
        self.training_map = training_map
        self.testing_map = testing_map

    def match_closest_result(self,key_item:str):
        test_item = self.testing_map[key_item]
        closest_prediciont = ['',100.0]
        res_ = []
        test_aggregated = [np.mean(np.array(train_gesture), axis=0) for train_gesture in test_item]

        for values in range(len(test_item)):
            max_diff = 10000.0
            prediction_str = ''
            for keys in self.training_map.keys():
                self.__get_y(self.training_map[keys][values],test_item[values])
                self.__cosine_similarity()
                test_val= min(self.cosine_sim.min(),max_diff)
                if test_val < max_diff:
                    max_diff = test_val
                    prediction_str= keys
            res_.append((prediction_str,max_diff))
        return res_
    def individual_testing(self,key_item,test_item):
        train= self.training_map[key_item]
        test = self.testing_map[test_item]
        # test_aggregated = [np.mean(np.array(tester),axis=0) for tester in test]
        # for values in range(len(test)):
        similar_test_gesture = [100.0]
        tested = 0
        print('Hello')
        for gestures in test:
            for trained in train:
                tested+=1
                print('Hello')
                self.__get_y(trained,gestures)
                self.__cosine_similarity()
                similar_test_gesture = min(similar_test_gesture,self.cosine_sim)
        print(similar_test_gesture,tested)

    def __get_y(self,training_data,testing_data):
        y_true = tf.convert_to_tensor(training_data)
        y_pred = tf.convert_to_tensor(testing_data)
        self.y_true_normalize = tf.nn.l2_normalize(y_true,axis=-1)
        self.y_pred_normalize = tf.nn.l2_normalize(y_pred,axis=-1)
    def __cosine_similarity(self):
        self.cosine_sim = tf.reduce_sum(tf.multiply(self.y_true_normalize,self.y_pred_normalize), axis = 0).numpy() #returns the prediction result

classifier = Classifier(training_set,testing_set)
# res = classifier.match_closest_result('H-DecreaseFanSpeed')

results_array = []

for tests in testing_set.keys():
    results_array.append((tests,classifier.match_closest_result(tests)))
# classifier.individual_testing('H-DecreaseFanSpeed','H-DecreaseFanSpeed')
for i in results_array:
    print(i)




# %%
import pandas as pd
res_array = []
for values in results_array:
    for tester in values[1::2]:
        for vals in tester:
            res_array.append(vals[1])

        # print(tester)
        # print(tester[1])
df = pd.DataFrame(res_array)
df.to_csv('Results.csv')

# %%
# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================



# y_true = tf.convert_to_tensor(train_speed,dtype=tf.float32)
# y_pred = tf.convert_to_tensor(test_speed,dtype=tf.float32)



# y_true_normalized = tf.nn.l2_normalize(y_true,axis=-1)
# y_pred_normalized = tf.nn.l2_normalize(y_pred,axis=-1)

# cosine_similarity = tf.reduce_sum(tf.multiply(y_true_normalized, y_pred_normalized), axis=1)

# print(cosine_similarity.numpy())




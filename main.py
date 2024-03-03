# %%
import pandas as pd
import os
from helper__lib.training_classifier import Framer
from helper__lib.label_mapper import Label_mapper, Training_map
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from helper__lib.classifier import Classifier
from hfe import HandShapeFeatureExtractor as HFe2
import time

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
print('Starting gathering Training data frames')

start_ = time.perf_counter_ns()
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
end_ = time.perf_counter_ns()
print('Recorded training frames')
print(f'Operation took  {end_-start_}')
# %%

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video
start_ = time.perf_counter_ns()
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
    result = frame_getter.get_frames_no_labels(videos,counter)
    test_frames.append(result)
end_ = time.perf_counter_ns()
print('Recorded test frames')
print(f'Operation took {end_-start_}')
# %%


print('Getting predictions now')
pred = HFe2().get_instance()

training_sample = {}

for key,values in predictionMap.items():
    for frames in values:
        prediction = pred.get_instance().extract_feature(frames)
        if key not in training_sample:
            training_sample[key] = [prediction]
        else:
            training_sample[key].append(prediction)

testing_sample = []
for a,values in enumerate(test_frames):
    testing_sample.append(pred.get_instance().extract_feature(values))
print('Gathered predictions')
# %%

# %%
# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
print('Classifing data')
classifier = Classifier(training_sample)
results_array = []

for tests in testing_sample:
    results_array.append(classifier.match_closest_result(tests))

#######################################



print('Recording Data to CSV')
result_mapper = Label_mapper()
result_list = []
for values in results_array:
    res = result_mapper.get_result(values[1])
    result_list.append(res)

# --------------------------------------------------------
#               Write to csv file predicitions             
# --------------------------------------------------------

df = pd.DataFrame(result_list)
df.to_csv('Results.csv',index=False,header=False)
print('Thank you - Project done by Mateo Ortega ')


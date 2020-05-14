# from pyAudioAnalysis.pyAudioAnalysis import audioBasicIO
# from pyAudioAnalysis.pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import os
import tqdm
import numpy as np
import time
import math
from pydub import AudioSegment
import random
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score

# Feature extractor template:
# Input: path_to_file
# Output: feature_vec
# Error Checking: return None

# FEATURE EXTRACTOR 1
def pyaudioextraction(path, fs_factor, overlap_factor, stereo=False):
    try:
        [Fs, x] = audioBasicIO.read_audio_file(path)
        if stereo:
            x = audioBasicIO.stereo_to_mono(x)
        F, f_names = ShortTermFeatures.feature_extraction(x, Fs, fs_factor * Fs, overlap_factor * Fs)
        return F.T.flatten()
    except:
        return None

# Main Functions
def randomize(dataset, labels):
    permutation = np.random.permutation(dataset.shape[0])
    shuffled_matrix = dataset[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_matrix, shuffled_labels

def extract(fs, overlap, extractor=pyaudioextraction, shuffle=True):
    matrix = []
    labels = []

    stutter_files = os.listdir('../outputs_stutter')
    nonstutter_files = os.listdir('../outputs_nonstutter')

    for i, file in enumerate(tqdm.tqdm(stutter_files + nonstutter_files)):
        if i < len(stutter_files):
            fpath = os.path.join('../outputs_stutter', file)
            output_vec = extractor(fpath, fs, overlap)

            if output_vec is not None:
                matrix.append(output_vec)
                labels.append(1.0)
        else:
            fpath = os.path.join('../outputs_nonstutter', file)
            output_vec = extractor(fpath, fs, overlap)

            if output_vec is not None:
                matrix.append(output_vec)
                labels.append(0.0)

    matrix = np.array(matrix)
    labels = np.array(labels)

    if shuffle:
        matrix, labels = randomize(matrix, labels)

    return matrix, labels

def write_files(matrix, labels):
    with open('vectors.txt', 'w') as vectors:
        for vec in matrix:
            vectors.write(",".join(str(c) for c in vec)+'\n')

    with open('labels.txt', 'w') as labs:
        for label in labels:
            labs.write(str(label)+'\n')

'''
def train_and_score(X, y):
    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    score = np.mean(cross_val_score(clf, X, y, cv=5))
    return score
'''

if __name__ == '__main__':
    inputs, labels = extract(0.78, 0.38)
    # print(train_and_score(inputs, labels))

    # print(grid_search())
    # print(reduce_fs(start_fs = 0.7887469810790151))

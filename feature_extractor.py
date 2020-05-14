from pyAudioAnalysis.pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis.pyAudioAnalysis import ShortTermFeatures
import os
import tqdm
import numpy as np
import time
import math
from pydub import AudioSegment
# import librosa

# Feature extractor template:
# Input: path_to_file
# Output: feature_vec
# Error Checking: return None

# FEATURE EXTRACTOR 1
def pyaudioextraction(path):
    try:
        [Fs, x] = audioBasicIO.read_audio_file(path)
        F, f_names = ShortTermFeatures.feature_extraction(x, Fs, Fs, Fs)
        return F.T.flatten()
    except:
        return None

# FEATURE EXTRACTOR 2
'''
def librosa(path):
    try:
        y, sr = librosa.load(path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr).flatten()
        return mfccs
    except:
        return None
'''

# Main Functions
def randomize(dataset, labels):
    permutation = np.random.permutation(dataset.shape[0])
    shuffled_matrix = dataset[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_matrix, shuffled_labels

def extract(extractor, shuffle=True):
    matrix = []
    labels = []

    stutter_files = os.listdir('outputs_stutter_1')
    nonstutter_files = os.listdir('outputs_nonstutter_1')

    for i, file in enumerate(tqdm.tqdm(stutter_files + nonstutter_files)):
        if i < len(stutter_files):
            fpath = os.path.join('outputs_stutter_1', file)
            output_vec = extractor(fpath)

            if output_vec is not None:
                matrix.append(output_vec)
                labels.append(1.0)
        else:
            fpath = os.path.join('outputs_nonstutter_1', file)
            output_vec = extractor(fpath)

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

if __name__ == '__main__':
    # inputs, labels = extract(librosa)
    inputs, labels = extract(pyaudioextraction)
    write_files(inputs, labels)

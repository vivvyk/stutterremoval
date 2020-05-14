from pydub import AudioSegment
# import segmentation
import pandas as pd
import math
import os
import time
import pickle
# import feature_extractor_GS
# import GS
# from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
# from pyAudioAnalysis import audioBasicIO
# from pyAudioAnalysis import ShortTermFeatures
import csv
import numpy as np

def process_probs(preds, LOWERTHRESHOLD=0.5, UPPERTHRESHOLD=1.0, inclusive=False, maxcount=14):
    processed_preds = {}
    count = 0
    for key in preds.keys():
        if count >= maxcount:
            break

        probs = preds[key]
        if UPPERTHRESHOLD >= probs[0] >= LOWERTHRESHOLD:
            processed_preds[key] = 1.0
            count += 1
        else:
            processed_preds[key] = 0.0

    return processed_preds

def get_stutter_frame_times(frame_preds, frame_time, overlap):
    stutter_times = {}
    for key in frame_preds.keys():
        if frame_preds[key] == 1:
            start = (frame_time - overlap) * key * 1000
            end = ((frame_time - overlap) * key + frame_time) * 1000
            stutter_times[key] = (round(start, 2), round(end, 2))

    return stutter_times

def correct_audio(og_audio, stutter_times):

    audio = AudioSegment.from_wav(og_audio)
    length = int(math.ceil(len(audio)/1000))

    correct_audio = None

    start_time = 0
    end_time = 0
    for key in stutter_times.keys():
        start_stut, end_stut = stutter_times[key]
        end_time = start_stut

        chunk = audio[start_time:end_time]

        if correct_audio is None:
            correct_audio = chunk
        else:
            if len(chunk) == 0: crossfade = 0
            else: crossfade = 50
            correct_audio = correct_audio.append(chunk, crossfade=crossfade)

        start_time = end_stut

    if correct_audio is None:
        correct_audio = audio
    else:
        correct_audio = correct_audio.append(audio[start_time:length*1000], crossfade=crossfade)

    correct_audio.export("corrected.wav", format="wav")

if __name__ == '__main__':
    frame, overlap = 0.50, 0.0
    frame_factor, overlap_factor = 0.5, 0.29

    base = './probabilities'
    for f in os.listdir(base):
        probs = {}
        path = os.path.join(base, f)
        for i, row in enumerate(open(path, 'r')):
            probs[i] = np.fromstring(row.strip('\n')[1:-1], dtype=float, sep=' ')

        predictions = process_probs(probs, LOWERTHRESHOLD=0.54, UPPERTHRESHOLD=0.77)
        stutter_times = get_stutter_frame_times(predictions, frame, overlap)
        correct_audio("ray_demnitz.wav", stutter_times)

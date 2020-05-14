from pydub import AudioSegment
import segmentation
import math
import os
import time
import pickle
import feature_extractor_GS
import GS
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import csv
from numpy import array

def train_clf(frame, overlap, frame_factor, overlap_factor):
    GS.write_frames(frame, overlap)
    inputs, labels = feature_extractor_GS.extract(frame_factor, overlap_factor)
    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf.fit(inputs, labels)
    return clf

def write_out_audio(audio, length, frame_time, overlap):
    # Turn audio into chunk

    curr_time = 0.0
    while curr_time < length:

        curr_time = round(curr_time, 2)
        t1 = curr_time * 1000
        t2 = (curr_time+frame_time) * 1000

        outfile = os.path.join('../test_audio_frames', 'interval&' + str(time.time()) + '.wav')

        chunk = audio[t1:t2]
        chunk.export(outfile, format="wav")
        curr_time += frame_time - overlap


def open_and_process_audio(audio_file, frame, overlap):
    GS.clear_directory('../test_audio_frames')

    audio = AudioSegment.from_wav(audio_file)
    length = int(math.ceil(len(audio)/1000))

    write_out_audio(audio, length, frame, overlap)


def find_stutters(model, frame_factor, overlap_factor):
    base = '../test_audio_frames'
    preds = {}
    for i, f in enumerate(os.listdir(base)):
        path = os.path.join(base, f)
        features = feature_extractor_GS.pyaudioextraction(path, frame_factor, overlap_factor, stereo=True)
        try:
            prediction = model.predict_proba(features.reshape(1, -1))
            preds[i] = prediction
        except:
            pass

    return preds


def process_probs(preds, LOWERTHRESHOLD=0.5, UPPERTHRESHOLD=1.0):
    processed_preds = {}
    for key in preds.keys():
        probs = preds[key]
        if UPPERTHRESHOLD > probs[0][0] > LOWERTHRESHOLD:
            processed_preds[key] = 1.0
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
        crossfade = 0

        if correct_audio is None:
            correct_audio = chunk
        else:
            correct_audio = correct_audio.append(chunk, crossfade=crossfade)

        start_time = end_stut

    if correct_audio is None:
        correct_audio = audio
    else:
        correct_audio = correct_audio.append(audio[start_time:length*1000], crossfade=crossfade)

    correct_audio.export("corrected.wav", format="wav")


def calculate_ideal_thresholds(predict_probs, frame, overlap):
    stutters = [(1,2), (2,3), (3,3.5), (5.5,9.5), (12,13)]
    for key in predict_probs.keys():
        fid = key * (frame - overlap)
        if len(segmentation.is_overlap(stutters, (fid,fid+frame))) != 0:
            print("STUTTER: " + str(predict_probs[key]))
        else:
            print("NON-STUTTER: " + str(predict_probs[key]))





if __name__ == '__main__':
    # 0.59, 0.29

    frame, overlap = 0.50, 0.0
    frame_factor, overlap_factor = 0.5, 0.29

    #open_and_process_audio("/home/vivekkumar/SLP_Thesis/refactored_SLP/ray_demnitz.wav", frame, overlap)
    #clf = train_clf(frame, overlap, frame_factor, overlap_factor)
    #predictions_probs = find_stutters(clf, frame_factor, overlap_factor)

    predictions_probs = {0: array([[0.51, 0.49]]), 1: array([[0.41, 0.59]]), 2: array([[0.65, 0.35]]), 3: array([[0.65, 0.35]]), 4: array([[0.77, 0.23]]), 5: array([[0.59, 0.41]]), 6: array([[0.55, 0.45]]), 7: array([[0.52, 0.48]]), 8: array([[0.82, 0.18]]), 9: array([[0.83, 0.17]]), 10: array([[0.65, 0.35]]), 11: array([[0.77, 0.23]]), 12: array([[0.66, 0.34]]), 13: array([[0.75, 0.25]]), 14: array([[0.57, 0.43]]), 15: array([[0.71, 0.29]]), 16: array([[0.74, 0.26]]), 17: array([[0.54, 0.46]]), 18: array([[0.56, 0.44]]), 19: array([[0.78, 0.22]]), 20: array([[0.84, 0.16]]), 21: array([[0.82, 0.18]]), 22: array([[0.54, 0.46]]), 23: array([[0.53, 0.47]]), 24: array([[0.56, 0.44]]), 25: array([[0.66, 0.34]]), 26: array([[0.57, 0.43]]), 27: array([[0.59, 0.41]]), 28: array([[0.84, 0.16]]), 29: array([[0.48, 0.52]])}

    # Write prediction probs
    filename = "../probabilities/output-probabilities-" + str(time.time()) + ".csv"
    out_file = open(filename, "w")

    writer = csv.writer(out_file)
    for key, value in predictions_probs.items():
        writer.writerow(value)


    # predictions = process_probs(predictions_probs)
    # stutter_times = get_stutter_frame_times(predictions, frame, overlap)
    # print(stutter_times)
    # calculate_ideal_thresholds(predictions_probs, frame, overlap)
    # correct_audio("/home/vivekkumar/SLP_Thesis/refactored_SLP/ray_demnitz.wav", stutter_times)

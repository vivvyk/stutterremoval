from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import os
import tqdm
import numpy as np
import time
import math
from pydub import AudioSegment


# [Fs, x] = audioBasicIO.read_audio_file("/home/vivekkumar/refactored_SLP/data/data-Vamsi/M_0048_11y1m_1.wav")
# F, f_names = ShortTermFeatures.feature_extraction(x, Fs, Fs, Fs)
# plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0])
# plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()


def proc_dir_struct(dirname, frame_time=1, overlap=0.5):
    audio_dir = os.path.join(dirname, 'audios')
    stutter_dir = os.path.join(dirname, 'stutters')

    for filebase in [f.split(".")[0] for f in os.listdir(audio_dir)]:
        audio_file = os.path.join(audio_dir, filebase + '.wav')
        csv_file = os.path.join(stutter_dir, filebase + '.csv')

        audio = AudioSegment.from_wav(audio_file)
        length = int(math.ceil(len(audio)/1000))
        stutters = process_transcription(csv_file, length, frame_time)
        process_audio(audio, length, stutters, frame_time, overlap)


def is_overlap(tuples, search):
    res = []
    for t in tuples:
        if(t[1]>search[0] and t[0]<search[1]):
            res.append(t)
    return res

def process_transcription(filename, length, frame_time):
    # Process transcription file

    stutter_intervals = []
    # Process file
    with open(filename, 'r') as f:
        for line in f:
            sline = line.strip().strip('"').split(",")
            stutter_intervals.append((float(sline[0].strip()), float(sline[1].strip())))


    # stutters = []
    # for i in range(length):
    #    if len(overlap(stutter_intervals, (i*frame_time, (i*frame_time)+frame_time))) != 0:
    #        stutters.append(round(i*frame_time, 2))

    # print(overlap(stutter_intervals, (8, 9)))
    return stutter_intervals

def process_audio(audio, length, stutters, frame_time, overlap):
    # Turn audio into chunks
    curr_time = 0.0
    while curr_time < length:

        curr_time = round(curr_time, 2)
        t1 = curr_time * 1000
        t2 = (curr_time+frame_time) * 1000
        if len(is_overlap(stutters, (curr_time, curr_time+frame_time))) != 0:
            outfile = os.path.join('../outputs_stutter', 'interval&' + str(time.time()) + '.wav')
        else:
            outfile = os.path.join('../outputs_nonstutter', 'interval&' + str(time.time()) + '.wav')

        chunk = audio[t1:t2]
        chunk.export(outfile, format="wav")
        curr_time += frame_time - overlap

if __name__ == '__main__':
    proc_dir_struct('/home/vivekkumar/SLP_Thesis/refactored_SLP/data/data-Kevin', frame_time=0.98, overlap=0.90)
    proc_dir_struct('/home/vivekkumar/SLP_Thesis/refactored_SLP/data/data-Vamsi', frame_time=0.98, overlap=0.90)

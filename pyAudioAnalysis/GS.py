import segmentation
import os
import feature_extractor_GS
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
import pickle

def clear_directory(dir):
    filelist = [ f for f in os.listdir(dir) if f.endswith(".wav") ]
    for f in filelist:
        os.remove(os.path.join(dir, f))

def write_frames(frame, overlap):
    clear_directory('../outputs_stutter')
    clear_directory('../outputs_nonstutter')

    # segmentation.proc_dir_struct('/home/vivekkumar/SLP_Thesis/refactored_SLP/data/data-Kevin', frame_time=frame, overlap=overlap)
    segmentation.proc_dir_struct('/home/vivekkumar/SLP_Thesis/refactored_SLP/data/data-Vamsi', frame_time=frame, overlap=overlap)

def train_and_score(X, y, save=False):
    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    score = np.mean(cross_val_score(clf, X, y, cv=5))
    return score

def grid_search(REPEAT=10):

    scores = {}

    for i in range(REPEAT):
        print("ITERATION: {}".format(i))

        random_frame_size = round(np.random.uniform(low=0.5, high=1.0), 2)
        random_overlap = round(np.random.uniform(low=0.0, high=0.75) * random_frame_size, 2)
        print("FRAME SIZE: {}; OVERLAP: {}".format(random_frame_size, random_overlap))

        print("WRITING FILES")
        write_frames(random_frame_size, random_overlap)

        print("TRAIN AND TESTING")
        inputs, labels = feature_extractor_GS.extract(random_frame_size, random_overlap)
        score = train_and_score(inputs, labels)

        scores[(random_frame_size, random_overlap)] = score

    return scores


def lower_incrementally(start_fs=1, best_overlap=1, lowers=6, lower_interval=0.01):
    scores = {}

    sp = start_fs
    bo = best_overlap

    for i in range(lowers):
        print("ITERATION: {}".format(i))

        print("FRAME SIZE: {}".format(sp))

        print("WRITING FILES")
        write_frames(sp, bo)

        print("TRAIN AND TESTING")
        inputs, labels = feature_extractor_GS.extract(sp, bo)
        score = train_and_score(inputs, labels)

        scores[sp] = score

        sp -= lower_interval

    return scores

if __name__ == '__main__':
    # GS_result = grid_search()
    # print(GS_result)

    # lower_result = lower_incrementally(start_fs=0.59, best_overlap=0.29)
    # print(lower_result)

    clear_directory('../outputs_stutter')
    clear_directory('../outputs_nonstutter')

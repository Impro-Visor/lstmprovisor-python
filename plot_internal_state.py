import constants
import matplotlib
import matplotlib.pyplot as plt
import itertools
import sys
import numpy as np
import os
import argparse
plt.ion()

import custom_cmap
my_cmap = matplotlib.colors.ListedColormap(custom_cmap.test_cm.colors[::-1])

# probs = np.load('generation/dataset_10000_probs.npy')
# probs_jump = np.load('generation/dataset_10000_info_0.npy')
# probs_chord = np.load('generation/dataset_10000_info_1.npy')
# chosen = np.load('generation/dataset_10000_chosen.npy')
# chosen_map = np.eye(probs.shape[-1])[chosen]

def plot_note_dist(mat, name="", show_octaves=True):
    f = plt.figure(figsize=(20,5))
    f.canvas.set_window_title(name)
    plt.imshow(mat.T, origin="lower", interpolation="nearest", cmap=my_cmap)
    plt.xticks( np.arange(0,4*(constants.WHOLE//constants.RESOLUTION_SCALAR),(constants.QUARTER//constants.RESOLUTION_SCALAR)) )
    plt.xlabel('Time (beat/12)')
    plt.ylabel('Note')
    plt.colorbar()
    if show_octaves:
        for y in range(0,36,12):
            plt.axhline(y + 1.5, color='c')
    for x in range(0,4*(constants.WHOLE//constants.RESOLUTION_SCALAR),(constants.QUARTER//constants.RESOLUTION_SCALAR)):
        plt.axvline(x-0.5, color='k')
    for x in range(0,4*(constants.WHOLE//constants.RESOLUTION_SCALAR),(constants.WHOLE//constants.RESOLUTION_SCALAR)):
        plt.axvline(x-0.5, color='c')
    plt.show()

def plot_scalar(mat, name=""):
    f = plt.figure(figsize=(20,5))
    f.canvas.set_window_title(name)
    plt.bar(range(mat.shape[0]),mat,1)
    plt.xticks( np.arange(0,4*(constants.WHOLE//constants.RESOLUTION_SCALAR),(constants.QUARTER//constants.RESOLUTION_SCALAR)) )
    plt.xlabel('Time (beat/12)')
    plt.ylabel('Strength')
    for x in range(0,4*(constants.WHOLE//constants.RESOLUTION_SCALAR),(constants.QUARTER//constants.RESOLUTION_SCALAR)):
        plt.axvline(x, color='k')
    for x in range(0,4*(constants.WHOLE//constants.RESOLUTION_SCALAR),(constants.WHOLE//constants.RESOLUTION_SCALAR)):
        plt.axvline(x, color='c')
    plt.show()


def plot_all(folder, idx=0):
    probs = np.load(os.path.join(folder,'generated_probs.npy'))
    chosen_raw = np.load(os.path.join(folder,'generated_chosen.npy'))
    chosen = np.eye(probs.shape[-1])[chosen_raw]
    plot_note_dist(probs[idx], 'Probabilities')
    plot_note_dist(chosen[idx], 'Chosen')
    try:
        for i in itertools.count():
            probs_info = np.load(os.path.join(folder,'generated_info_{}.npy'.format(i)))
            if len(probs_info.shape) == 3:
                show_octaves = probs_info.shape[2] < 40
                plot_note_dist(probs_info[idx], 'Info {}'.format(i), show_octaves)
            else:
                plot_scalar(probs_info[idx], 'Info {}'.format(i))
    except FileNotFoundError:
        pass

parser = argparse.ArgumentParser(description='Plot the internal state of a network')
parser.add_argument('folder', help='Directory with the generated files')
parser.add_argument('idx', type=int, help='Zero-based index of the output to visualize')

if __name__ == '__main__':
    args = parser.parse_args()
    plot_all(**vars(args))
    input("Press enter to close.")

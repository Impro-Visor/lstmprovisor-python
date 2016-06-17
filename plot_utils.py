import constants
import matplotlib
import matplotlib.pyplot as plt
plt.ion()

# probs = np.load('generation/dataset_10000_probs.npy')
# probs_jump = np.load('generation/dataset_10000_info_0.npy')
# probs_chord = np.load('generation/dataset_10000_info_1.npy')
# chosen = np.load('generation/dataset_10000_chosen.npy')
# chosen_map = np.eye(probs.shape[-1])[chosen]


def plot_generated(mat):
    plt.figure()
    plt.imshow(mat.T, origin="lower", interpolation="nearest", cmap='viridis')
    plt.colorbar()
    for y in range(0,36,12):
        plt.axhline(y + 1.5, color='c')
    for x in range(0,4*(constants.WHOLE//constants.RESOLUTION_SCALAR),(constants.QUARTER//constants.RESOLUTION_SCALAR)):
        plt.axvline(x-0.5, color='k')
    for x in range(0,4*(constants.WHOLE//constants.RESOLUTION_SCALAR),(constants.WHOLE//constants.RESOLUTION_SCALAR)):
        plt.axvline(x-0.5, color='c')

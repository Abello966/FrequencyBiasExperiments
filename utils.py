import gc
import numpy as np
import cv2
from scipy import fftpack
from skimage import color

# normalize an image for range [0..1] for imshow
def normalize_image(Xfr):
    Xfr_norm = Xfr - np.min(Xfr)
    Xfr_norm = Xfr_norm / np.max(Xfr_norm)
    return Xfr_norm

# Calculate the accuracy of mod in Xdatagen using an optional
# preprocessing function preproc
def get_accuracy_iterator(mod, Xdatagen, preproc=lambda x: x):
    acc = 0
    npoints = 0
    for i in range(len(Xdatagen)):
        Xfr, yfr = next(Xdatagen)
        Xfr = preproc(Xfr)

        ypred = mod.predict(Xfr)
        yfr = np.argmax(yfr, axis=1)
        ypred = np.argmax(ypred, axis=1)

        npoints += len(yfr)
        acc += np.sum(yfr == ypred)

        del Xfr
        del yfr
        del ypred
        gc.collect()

    return acc / npoints

# Calculate the empirical distribution of energy throughout the frequency spectra
# of a dataset represented by Xfr (with shape (ex, width, height, channels))
def get_mean_energy_dataset(Xfr):
    avg_energy_fr = np.zeros(Xfr.shape[1:3])
    for i in range(Xfr.shape[0]):
        image = color.rgb2gray(Xfr[i])
        freq = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
        energy = cv2.magnitude(freq[:, :, 0], freq[:, :, 1])
        energy[0][0] = 0
        avg_energy_fr += energy
    avg_energy_fr /= Xfr.shape[0]
    avg_energy_fr = fftpack.fftshift(avg_energy_fr)
    return avg_energy_fr

def get_mean_energy_iterator(Xdatagen):
    avg_energy_fr = np.zeros(Xdatagen.image_shape[:-1])
    for i in range(len(Xdatagen)):
        Xfr, _ = next(Xdatagen)
        avg_energy_fr += get_mean_energy_dataset(Xfr)
    avg_energy_fr /= len(Xdatagen)
    return avg_energy_fr

# Get a theoretical model of the distribution of energy
def get_relevance_model(shape, decay, A=1):
    a = np.zeros((shape[0], shape[1]))
    cx, cy = shape[0] / 2, shape[1] / 2
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i][j] = abs(cy - i) + abs(cx - j)

    a[int(cx)][int(cy)] = 1
    relevance = A / (a ** decay)
    relevance[int(cx)][int(cy)] = 1 # nevermind this
    return relevance

# get binary mask that eliminates frequencies with l1 norm greater than radius1
# and lesser or equal radius2
def get_mask(image, radius1, radius2):
    rows, cols = image.shape[0:2]
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.ones((rows, cols), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = abs(x - center[0]) + abs(y - center[1]) > radius1
    mask_area *= abs(x - center[0]) + abs(y - center[1]) <= radius2
    mask[mask_area] = 0
    return mask

# get an energy histogram using step_list as a list of increasing integers
# in which each adjacent pair represents a disc
def get_energy_histogram(energy, step_list):
    height = list()
    width = list()
    for i in range(len(step_list) - 1) :
        radius1 = step_list[i]
        radius2 = step_list[i + 1]
        mask_area = get_mask(energy, radius1, radius2)
        mask_area = 1 - mask_area

        height.append(np.sum(energy * mask_area) / np.sum(energy))
        width.append(step_list[i+1] - step_list[i])
    return height, width

#given a real-valued relevance mask get how much this mask represents of total relevance
def percent_cut_relevance(mask, relevance):
    return np.sum((mask == 0) * relevance) / np.sum(relevance)

# given a relevance mask, return a list of frequencies in which each adjacent
# pair represents a disc with approximately percent of the total energy
def get_percentage_masks_relevance(relevance, percent):
    range_result = [0]
    last_result = 0
    for i in range(1, 100, int(percent * 100)):
        next_result = last_result
        mask = get_mask(relevance, np.inf, np.inf)

        while percent_cut_relevance(mask, relevance) < percent and next_result < (relevance.shape[0]):
            last_pct = percent_cut_relevance(mask, relevance)
            next_result = next_result + 1
            mask = get_mask(relevance, last_result, next_result)
            if abs(percent_cut_relevance(mask, relevance) - percent) > abs(last_pct - percent):
                next_result -= 1
                break

        range_result.append(next_result)
        last_result = next_result
    return range_result

## given an image which may or may not be RGB, eliminate all frequencies which belong to the
## disc defined by radius1 (not inclusive) and radius2 (inclusive)
def remove_frequency_ring(image, radius1, radius2, multicolor=True):
    image = image.astype(np.float32)

    if multicolor:
        imager = image[:, :, 0]
        imageg = image[:, :, 1]
        imageb = image[:, :, 2]
        imager_res = remove_frequency_ring(imager, radius1, radius2, multicolor=False)
        imageg_res = remove_frequency_ring(imageg, radius1, radius2, multicolor=False)
        imageb_res = remove_frequency_ring(imageb, radius1, radius2, multicolor=False)
        return np.stack([imager_res, imageg_res, imageb_res], axis=-1)

    rows, cols = image.shape[0:2]
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = get_mask(image, radius1, radius2)
    mask[crow][ccol] = 1

    freq = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
    freq = fftpack.fftshift(freq)

    freq[:, :, 0] = freq[:, :, 0] * mask
    freq[:, :, 1] = freq[:, :, 1] * mask

    freq = fftpack.ifftshift(freq)
    distorted = cv2.idft(freq, flags=cv2.DFT_SCALE + cv2.DFT_COMPLEX_INPUT)
    final = distorted[:, :, 0]
    return final

def remove_frequency_ring_dataset(dataset, radius1, radius2):
    return np.array([remove_frequency_ring(image, radius1, radius2, multicolor=True) for image in dataset])

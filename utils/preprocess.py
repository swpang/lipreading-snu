import cv2
import random
import numpy as np
from skimage.util import random_noise
SALT_PEPPER_RATIO = 0.0125
__all__ = ['Compose', 'Normalize', 'CenterCrop', 'RgbToGray', 'RandomCrop',
           'HorizontalFlip', 'AddNoise', 'NormalizeUtterance', 'Rotation', 'Noise_Image']


class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RgbToGray(object):
    """Convert image to grayscale.
    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a numpy.ndarray of shape (H x W x C) in the range [0.0, 1.0].
    """

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Image to be converted to gray.
        Returns:
            numpy.ndarray: grey image
        """
        frames = np.stack([cv2.cvtColor(_, cv2.COLOR_RGB2GRAY) for _ in frames], axis=0)
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)


class CenterCrop(object):
    """Crop the given image at the center
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w-tw)
        delta_h = random.randint(0, h-th)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class HorizontalFlip(object):
    """Flip image horizontally.
    """

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
                # cv2.imwrite("/home/deep_learning_46/image/flipped_{}.png".format(index), frames[index])
        return frames


class NormalizeUtterance():
    """Normalize per raw audio by removing the mean and divided by the standard deviation
    """
    def __call__(self, signal):
        signal_std = 0. if np.std(signal)==0. else np.std(signal)
        signal_mean = np.mean(signal)
        return (signal - signal_mean) / signal_std


class AddNoise(object):
    """Add SNR noise [-1, 1]
    """

    def __init__(self, noise, snr_levels=[-5, 0, 5, 10, 15, 20, 9999]):
        assert noise.dtype in [np.float32, np.float64], "noise only supports float data type"
        
        self.noise = noise
        self.snr_levels = snr_levels

    def get_power(self, clip):
        clip2 = clip.copy()
        clip2 = clip2 **2
        return np.sum(clip2) / (len(clip2) * 1.0)

    def __call__(self, signal):
        assert signal.dtype in [np.float32, np.float64], "signal only supports float32 data type"
        snr_target = random.choice(self.snr_levels)
        if snr_target == 9999:
            return signal
        else:
            # -- get noise
            start_idx = random.randint(0, len(self.noise)-len(signal))
            noise_clip = self.noise[start_idx:start_idx+len(signal)]

            sig_power = self.get_power(signal)
            noise_clip_power = self.get_power(noise_clip)
            factor = (sig_power / noise_clip_power ) / (10**(snr_target / 10.0))
            desired_signal = (signal + noise_clip*np.sqrt(factor)).astype(np.float32)
            return desired_signal

#########################################################################################

class Rotation(object):
    """
    rotate_ratio: probability of rotation being initiated
    """

    def __init__(self, rotate_ratio):
        self.rotate_ratio = rotate_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be rotated with a probability rotate_ratio
        Returns:
            numpy.ndarray: Rotated image.
        """
        t, h, w = frames.shape
        # print(frames.shape)

        angle = random.uniform(-30, 30)
        if random.random() < self.rotate_ratio:
            for index in range(t):
                image_center = tuple(np.array(frames[index].shape[1::-1]) / 2)
                rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
                frames[index] = cv2.warpAffine(frames[index], rot_mat, frames[index].shape[1::-1], flags=cv2.INTER_LINEAR)

                # print(frames[index])
                # cv2.imwrite("/home/deep_learning_46/image/rotated_{}.png".format(index), frames[index])
        return frames



class Noise_Image(object):
    """
    Add salt and pepper noise to image
    noise_ratio: probability of noise_ratio being initiated


    REMEMBER: This process should be done before the frame being NORMALIZED
    """

    def __init__(self, noise_ratio):
        self.noise_ratio = noise_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be Noised with a probability rotate_ratio
        Returns:
            numpy.ndarray: Noised image.
        """
        t, h, w = frames.shape

        if random.random() < self.noise_ratio:
            for index in range(t):
                frames[index] = self.salt_and_pepper_noise(frames[index])

                cv2.imwrite("/home/deep_learning_46/image/salt_pepper_{}.png".format(index), frames[index])
        return frames


    def salt_and_pepper_noise(self, frame, ratio = SALT_PEPPER_RATIO):
        dst = np.copy(frame)
        h, w = dst.shape
        for i in range(h):
            for j in range(w):
                rand_01 = random.random()

                if rand_01 < ratio:
                    dst[i][j] = 0
                elif rand_01 > 1-ratio:
                    dst[i][j] = 255
                # else, nothing. retaining pixel values.

        return dst








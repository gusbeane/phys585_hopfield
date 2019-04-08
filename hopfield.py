import numpy as np
from mnist import MNIST

import matplotlib.pyplot as plt

class hopfield(object):
    def __init__(self, train_data, test_data, theta=0.5, nprocess=1000):
        """Simple Hopfield neural network.

        Takes in training data and test data 

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            train_data (:obj:`list` of :obj:`float`): The training data as a (N, m)
                numpy array - N is the number of training samples and m is the
                length of the image.
            test_data (:obj:`list` of :obj:`float`): The testing data as a (N, m)
                numpy array - N is the number of testing samples and m is the
                length of the image.
            param3 (:obj:`list` of :obj:`str`): Description of `param3`.
        """
        self.train_data, self.test_data = train_data, test_data
        self._check_input_data_()

        self.w = self.train(self.train_data)
        self.processed_data = self.process(self.test_data, theta, nprocess)

    def train(self, train_data):
        w = np.zeros(np.shape(train_data)[1])
        for img in train_data:
            this_w = self._create_weight_(img)
            w = np.add(w, this_w)
        return w

    def process(self, test_data, theta, nprocess):
        processed_data = []
        for img in test_data:
            new_img = self._process_one_image_(img, theta, nprocess)
            processed_data.append(new_img)
        return processed_data

    def _process_one_image_(self, img, theta, nprocess):
        n = len(img)
        for _ in range(nprocess):
            i = np.random.randint(0, n-1)
            u = np.dot(self.w[i][:], img) - theta
            if u > 0:
                img[i] = 1
            else:
                img[i] = -1
        return img

    def _create_weight_(self, img):
        """Create weight matrix from image. 

        Args:
            img (:obj:`list` of :obj:`float`): The img of which to make a weight
                matrix. img must be a 1D numpy array
        """
        w = np.outer(img, img)
        np.fill_diagonal(w, 0)
        return w

    def _check_input_data_(self):
        for data in [self.train_data, self.test_data]:
            if len(np.shape(data)) == 1:
                data = np.reshape(data, (1, len(data)))

def choose_training(images, labels):
    k = np.array([1, 3, 5, 7, 2, 0, 13, 15, 17, 4])
    i = images[k]
    l = labels[k]

    return k, i, l

def choose_test(images, labels):
    k = np.array([200, 201, 202, 203])
    i = images[k]
    l = labels[k]

    return k, i, l


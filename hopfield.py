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

def massage_images(images):
    images = np.asarray(images)
    images = np.divide(images, 255)
    # slow, but I cant get a slicker solution to work for some reason
    for i in range(np.shape(images)[0]):
        for j in range(np.shape(images)[1]):
            if images[i][j] > 0.5:
                images[i][j] = 1
            else:
                images[i][j] = -1
    return images

if __name__ == '__main__':
    mndata = MNIST('samples')

    images, labels = mndata.load_training()
    images = np.asarray(images)
    labels = np.asarray(labels)
    
    train_keys, train_images, train_labels = choose_training(images, labels)
    test_keys, test_images, test_labels = choose_test(images, labels)

    train_images = massage_images(train_images)
    test_images = massage_images(test_images)

    for i in range(len(test_keys)):
        test_img = np.reshape(test_images[i], (28,28))

        plt.imshow(test_img)
        plt.savefig('images/input_test_'+str(i)+'.png')
        plt.close()

    h = hopfield(train_images, test_images, theta=0.5, nprocess=5000)

    for i in range(len(test_keys)):
        test_img = np.reshape(test_images[i], (28,28))
        proc_img = np.reshape(h.processed_data[i], (28,28))

        plt.imshow(test_img)
        plt.savefig('images/test_'+str(i)+'.png')
        plt.close()

        plt.imshow(proc_img)
        plt.savefig('images/proc_'+str(i)+'.png')
        plt.close()

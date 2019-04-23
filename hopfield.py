import numpy as np
from mnist import MNIST

import matplotlib.pyplot as plt

class hopfield(object):
    def __init__(self, train_data, test_data, theta=0.5, nprocess=1000, storkey=False):
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

        self.w = self.train(self.train_data, storkey=storkey)
        print(self.w[:10][:10])
        self.processed_data = self.process(self.test_data, theta, nprocess)

    def train(self, train_data, storkey=False):
        if storkey:
            return self._train_storkey_(train_data)

        i = np.shape(train_data)[1]
        npix = i**2
        w = np.zeros((i,i))
        for img in train_data:
            this_w = self._create_weight_(img)
            w = np.add(w, this_w)
        # w = w/len(train_data)
        w = w/npix
        print(np.shape(w))
        return w

    def _train_storkey_(self, train_data):
        l = np.shape(train_data)[1]
        w = np.zeros((l,l))

        for img in train_data:
            h = np.matmul(w, img)
            for i in range(len(img)):
                for j in range(len(img)):
                    if i != j:
                        w[i][j] += img[i]*img[j]
                        w[i][j] -= img[i]*h[j]
                        w[i][j] -= h[i]*img[j]

        w = w/l**2
        w = w/len(train_data)
        np.fill_diagonal(w, 0)
        return w

    def process(self, test_data, theta, nprocess):
        processed_data = []
        for img in test_data:
            new_img = self._process_one_image_(img, theta, nprocess)
            processed_data.append(new_img)
        return processed_data

    def _process_one_image_(self, img, theta, nprocess):
        myimg = np.copy(img)
        n = len(myimg)
        u_list = []
        e_old = self.energy(myimg, theta)
        print('new e:', e_old)
        for _ in range(round(nprocess/100)):
            for z in range(100):
                i = np.random.randint(0, n-1)
                u = np.dot(self.w[i][:], myimg) - theta
                myimg[i] = np.sign(u)
            e = self.energy(myimg, theta)
            print(e)
            if e == e_old:
                return myimg
            # u_list.append(u)
            # if u > 0:
            #     img[i] = 1
            # else:
            #     img[i] = -1
        return myimg

    def energy(self, img, theta):
        return -0.5 * img @ self.w @ img + np.sum(img * theta)

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

    # train_images = np.random.rand(*np.shape(train_images))*255

    train_images = massage_images(train_images)
    test_images = massage_images(test_images)

    for i in range(len(test_keys)):
        test_img = np.reshape(test_images[i], (28,28))

        plt.imshow(test_img)
        plt.savefig('images/input_test_'+str(i)+'.png')
        plt.close()

    for i in range(len(train_keys)):
        train_img = np.reshape(train_images[i], (28,28))

        plt.imshow(train_img)
        plt.savefig('images/train_'+str(i)+'.png')
        plt.close()

    h = hopfield(train_images[[1,3,8]], test_images, theta=0, nprocess=5000, storkey=False)

    retrain_img = h.process(train_images, 0, 5000)

    for i in range(len(retrain_img)):
        img = np.reshape(retrain_img[i], (28,28))

        plt.imshow(img)
        plt.savefig('images/retrain_'+str(i)+'.png')
        plt.close()

    for i in range(len(test_keys)):
        test_img = np.reshape(test_images[i], (28,28))
        proc_img = np.reshape(h.processed_data[i], (28,28))

        plt.imshow(test_img)
        plt.savefig('images/test_'+str(i)+'.png')
        plt.close()

        plt.imshow(proc_img)
        plt.savefig('images/proc_'+str(i)+'.png')
        plt.close()

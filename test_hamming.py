import numpy as np
from mnist import MNIST

import matplotlib.pyplot as plt
from hopfield import hopfield, massage_images
from tqdm import tqdm

def choose_training(images, labels):
    k = np.array([1, 3, 5, 7, 2, 0, 13, 15, 17, 4])
    i = images[k]
    l = labels[k]

    return k, i, l

def choose_test(images, labels):
    k = np.arange(200, 400)
    i = images[k]
    l = labels[k]

    return k, i, l

def hamming_normalized(img1, img2):
    assert len(np.shape(img1)) == 1, "image is not 1d!"
    assert np.shape(img1) == np.shape(img2), "Images are not the same shape!"

    n = len(img1)
    k = len(np.where(img1 == img2)[0])

    return 1 - k/n

if __name__ == '__main__':
    mndata = MNIST('samples')

    images, labels = mndata.load_training()
    images = np.asarray(images)
    labels = np.asarray(labels)
    
    # these are from this script, not hopfield.py
    train_keys, train_images, train_labels = choose_training(images, labels) 
    test_keys, test_images, test_labels = choose_test(images, labels)

    train_images = massage_images(train_images)
    test_images = massage_images(test_images)

    # first do hebbian
    Ntrain_list = np.arange(2, 10)
    z = np.zeros(np.shape(Ntrain_list))
    to_plot = np.transpose([Ntrain_list, z, z])

    train_keys_to_choose = list(range(len(train_images)))

    for Ndx, Ntrain in enumerate(Ntrain_list):
        d = 0
        for _ in tqdm(range(20)):
            ktrain = np.random.choice(train_keys_to_choose, size=Ntrain, replace=False)
            ktest = np.where(np.isin(test_labels, train_labels[ktrain]))[0]

            d += len(ktest)

            h_hebb = hopfield(train_images[ktrain], test_images[ktest], theta=0, nprocess=2500)
            h_pinv = hopfield(train_images[ktrain], test_images[ktest], theta=0, nprocess=2500, pseudoinverse=True)

            tlabels = test_labels[ktest]
            corresponding_train = train_images[tlabels]

            proc_hebb = h_hebb.processed_data
            proc_pinv = h_pinv.processed_data

            for proc_img_hebb, proc_img_pinv, train_img in zip(proc_hebb, proc_pinv, corresponding_train):
                hd = hamming_normalized(proc_img_hebb, train_img)
                to_plot[Ndx][1] += hd
                hd = hamming_normalized(proc_img_pinv, train_img)
                to_plot[Ndx][2] += hd

        to_plot[Ndx][1] /= d
        to_plot[Ndx][2] /= d

    fig, ax = plt.subplots(1, 1)
    ax.plot(to_plot[:,0], to_plot[:,1], label='hebbian')
    ax.plot(to_plot[:,0], to_plot[:,2], label='pseudo inverse')

    ax.set_xlabel('N train')
    ax.set_ylabel('average normalized hamming distance')

    ax.legend()
    fig.tight_layout()
    fig.savefig('hamming.png')
    
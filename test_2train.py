import numpy as np
from mnist import MNIST

import matplotlib.pyplot as plt
from hopfield import hopfield, choose_training, choose_test, massage_images

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
        plt.savefig('images_2train/input_test_'+str(i)+'.png')
        plt.close()

    for i in range(len(train_keys)):
        train_img = np.reshape(train_images[i], (28,28))

        plt.imshow(train_img)
        plt.savefig('images_2train/train_'+str(i)+'.png')
        plt.close()

    h = hopfield(train_images[[1,3]], test_images, theta=0, nprocess=5000, storkey=False)

    retrain_img = h.process(train_images, 0, 5000)

    for i in range(len(retrain_img)):
        img = np.reshape(retrain_img[i], (28,28))

        plt.imshow(img)
        plt.savefig('images_2train/retrain_'+str(i)+'.png')
        plt.close()

    for i in range(len(test_keys)):
        test_img = np.reshape(test_images[i], (28,28))
        proc_img = np.reshape(h.processed_data[i], (28,28))

        plt.imshow(test_img)
        plt.savefig('images_2train/test_'+str(i)+'.png')
        plt.close()

        plt.imshow(proc_img)
        plt.savefig('images_2train/proc_'+str(i)+'.png')
        plt.close()
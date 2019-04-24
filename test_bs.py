import numpy as np

import matplotlib.pyplot as plt
from hopfield import hopfield, choose_training, choose_test, massage_images

if __name__ == '__main__':

    img1 = np.full((28,28), -1)
    np.fill_diagonal(img1, 1)

    img2 = np.full((28,28), -1)
    np.fill_diagonal(img2,1)
    img2 = np.fliplr(img2)

    img3 = np.full((28,28), -1)
    img3[0,:] = img3[:,0] = img3[:,-1] = img3[-1,:] = 1

    img1 = np.reshape(img1, (28*28,))
    img2 = np.reshape(img2, (28*28,))
    img3 = np.reshape(img3, (28*28,))

    train_images = np.array([img1, img2, img3])
    test_images = np.copy(train_images)

    for i in range(len(test_images)):
        test_img = np.reshape(test_images[i], (28,28))

        plt.imshow(test_img)
        plt.savefig('images_bs/input_test_'+str(i)+'.png')
        plt.close()

    for i in range(len(train_images)):
        train_img = np.reshape(train_images[i], (28,28))

        plt.imshow(train_img)
        plt.savefig('images_bs/train_'+str(i)+'.png')
        plt.close()

    h = hopfield(train_images, test_images, theta=0, nprocess=5000, storkey=False)

    retrain_img = h.process(train_images, 0, 5000)

    for i in range(len(retrain_img)):
        img = np.reshape(retrain_img[i], (28,28))

        plt.imshow(img)
        plt.savefig('images_bs/retrain_'+str(i)+'.png')
        plt.close()

    for i in range(len(test_images)):
        test_img = np.reshape(test_images[i], (28,28))
        proc_img = np.reshape(h.processed_data[i], (28,28))

        plt.imshow(test_img)
        plt.savefig('images_bs/test_'+str(i)+'.png')
        plt.close()

        plt.imshow(proc_img)
        plt.savefig('images_bs/proc_'+str(i)+'.png')
        plt.close()
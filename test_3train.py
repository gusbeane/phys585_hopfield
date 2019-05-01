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
        plt.savefig('images_3train/input_test_'+str(i)+'.png')
        plt.close()

    for i in range(len(train_keys)):
        train_img = np.reshape(train_images[i], (28,28))

        plt.imshow(train_img)
        plt.savefig('images_3train/train_'+str(i)+'.png')
        plt.close()

    h = hopfield(train_images[[1,3,8]], test_images, theta=0, nprocess=5000, storkey=False)

    retrain_img = h.process(train_images, 0, 5000)

    for i in range(len(retrain_img)):
        img = np.reshape(retrain_img[i], (28,28))

        plt.imshow(img)
        plt.savefig('images_3train/retrain_'+str(i)+'.png')
        plt.close()

    for i in range(len(test_keys)):
        test_img = np.reshape(test_images[i], (28,28))
        proc_img = np.reshape(h.processed_data[i], (28,28))

        plt.imshow(test_img)
        plt.savefig('images_3train/test_'+str(i)+'.png')
        plt.close()

        plt.imshow(proc_img)
        plt.savefig('images_3train/proc_'+str(i)+'.png')
        plt.close()

    img, tlist, elist = h._process_one_image_(test_images[0], theta=0, nprocess=5000, return_energy=True)

    plt.plot(tlist, elist)
    plt.xlabel('iteration')
    plt.ylabel('energy')
    plt.tight_layout()
    plt.savefig('images_3train/proc_0_energy_vs_t.png')

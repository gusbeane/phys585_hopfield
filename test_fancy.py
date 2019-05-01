import numpy as np
import skimage.data
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.filters import threshold_mean

import matplotlib.pyplot as plt
from hopfield import hopfield, choose_training, choose_test, massage_images

def preprocessing(img, w=128, h=128):
    # Resize image
    img = resize(img, (w,h), mode='reflect')

    # Thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1 # Boolian to int

    # Reshape
    flatten = np.reshape(shift, (w*h))
    return flatten

if __name__ == '__main__':

    camera = skimage.data.camera()
    astronaut = rgb2gray(skimage.data.astronaut())
    coffee = rgb2gray(skimage.data.coffee())
    moon = skimage.data.moon()
    an = skimage.data.microaneurysms()
    hdf = rgb2gray(skimage.data.hubble_deep_field())
    pg = skimage.data.page()
    rocket = rgb2gray(skimage.data.rocket())

    # Marge data
    data = [camera, astronaut, coffee, moon]

    # Preprocessing
    print("Start to data preprocessing...")
    data = [preprocessing(d) for d in data]

    train_images = np.array(data)
    test_images = np.copy(train_images)

    # randomly flip 20% of pixels
    for i in range(len(test_images)):
        img = test_images[i]

        nreplace = round(0.4*len(img))

        k = np.random.choice(list(range(len(img))), nreplace, replace=False)
        
        to_replace = np.random.randint(2, size=nreplace)
        to_replace = 2*to_replace - 1
        img[k] = to_replace

    for i in range(len(test_images)):
        test_img = np.reshape(test_images[i], (128,128))

        plt.imshow(test_img)
        plt.savefig('images_fancy/input_test_'+str(i)+'.png')
        plt.close()

    for i in range(len(train_images)):
        train_img = np.reshape(train_images[i], (128,128))

        plt.imshow(train_img)
        plt.savefig('images_fancy/train_'+str(i)+'.png')
        plt.close()

    h = hopfield(train_images, test_images, theta=0, nprocess=200000, storkey=False)

    retrain_img = h.process(train_images, 0, 5000)

    for i in range(len(retrain_img)):
        img = np.reshape(retrain_img[i], (128,128))

        plt.imshow(img)
        plt.savefig('images_fancy/retrain_'+str(i)+'.png')
        plt.close()

    for i in range(len(test_images)):
        test_img = np.reshape(test_images[i], (128,128))
        proc_img = np.reshape(h.processed_data[i], (128,128))

        plt.imshow(test_img)
        plt.savefig('images_fancy/test_'+str(i)+'.png')
        plt.close()

        plt.imshow(proc_img)
        plt.savefig('images_fancy/proc_'+str(i)+'.png')
        plt.close()

import cv2
import numpy as np
import argparse
import time
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

#img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
#img = cv2.imread("amostra01.jpg")
img = load_img('Negativa/c5/negativa06.jpg')
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='ruler', save_format='jpeg'):
    i += 1
    if i > 40:
        break  # otherwise the generator would loop indefinitely

lista = np.array(img)

print ("Imagem total: "+ str(x.shape))
print ("Altura (height): %d pixels" % (x.shape[1]))
print ('Largura (width): %d pixels' % (x.shape[2]))
print ('Canais (channels): %d'      % (x.shape[3]))

#blob = cv2.dnn.blobFromImage(img, 1, (34, 34), x.shape)
print("[INFO] loading model...")

#https://imasters.com.br/back-end/classificacao-de-imagens-com-deep-learning-e-tensorflow

#cv2.waitKey(0)
cv2.destroyAllWindows()

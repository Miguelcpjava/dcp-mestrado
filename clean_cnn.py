import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#https://gist.github.com/fchollet

train_data_dir = 'data/train' #pastas com as imagens de treinamento ??? QUal imagem eu coloco na pasta, as de 50x50 ou as das réguas originais
validation_data_dir = 'data/validation' #pastas com as imagens de de validação, as imagens reais
test_data_dir = 'data/test'
nb_train_samples = 1000
nb_validation_samples = 220
epochs = 20
batch_size = 16

# dimensions of our images of validation.
img_width, img_height = 50, 50

#img_training_width, img_training_height = 1040, 585

#Gerando as imagens
print('===============TREINAMENTO===============')
train_batches = ImageDataGenerator().flow_from_directory(train_data_dir, target_size=(img_width,img_height), classes=['positivo','negativo'],batch_size=batch_size)
print('=========================================')
print('================VALIDAÇÃO================')
validation_batches = ImageDataGenerator().flow_from_directory(validation_data_dir, target_size=(img_width,img_height), classes=['positivo','negativo'],batch_size=batch_size)
print('=========================================')
print('==================TESTES=================')
print('AINDA NÃO ESTÁ DISPONÍVEL................')
print('=========================================')
#img = load_img(train_data_dir+'positiva/layer1')
#x = img_to_array(img)  # this is a Numpy array with shape (3, 50, 50)
#x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 50, 50)

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(50, 50, 3)))
model.add(Activation('relu'))
#model.reshape((1,)+model.shape)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.summary()

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(units = 64))
model.add(Activation('relu'))
model.add(Dropout(0.5)) #Zerando 50 porcento da entrada

model.add(Dense(units = 128))
model.add(Activation('relu'))
model.add(Dropout(0.5)) #Zerando 50 porcento da entrada

model.add(Dense(units = 2)) # Duas unidades pelo fato que eu tenho duas classes
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',#optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(
        train_batches,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_batches,
        validation_steps=nb_validation_samples // batch_size)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('second_try.h5')  # always save your weights after training or during training
print('Fim do processamento')
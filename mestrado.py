import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import backend as K
from keras import metrics
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.metrics import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, image

train_data_dir = 'data/train' 
validation_data_dir = 'data/validation' 
test_data_dir = 'data/test'
nb_train_samples = 1000
nb_validation_samples = 36
epochs = 20
batch_size = 5 #Número de classes
# dimensions of our images of validation.
img_width, img_height = 50, 50

adam = Adam(lr=0.0005)#Adam(lr=0.001)
rmsprop = RMSprop() #RMSprop(lr=0.0005)

#Iniciando a construção da rede convolucional
#Criando o modelo
classifier = Sequential()
# 1st Layer
classifier.add(Conv2D(64, (3, 3), input_shape = (img_width, img_height, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(1,1)))
#classifier.add(BatchNormalization())
#classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
#O dropout consiste em configurar aleatoriamente uma taxa de fração de unidades de entrada para 0 
#em cada atualização durante o tempo de treinamento, o que ajuda a prevenir o overfitting.
classifier.add(Dropout(0.15))
classifier.add(Flatten())
#Adicionando as camadas
#1 fully connected layers
classifier.add(Dense(units = 80, activation = 'relu'))
classifier.add(Dropout(0.25))
#2 fully connected layers
classifier.add(Dense(units = 40, activation = 'relu'))
classifier.add(Dropout(0.25))
#3 fully connected layers
classifier.add(Dense(units = 5, activation = 'softmax'))
classifier.compile(optimizer = rmsprop, loss = 'categorical_crossentropy', metrics = ['accuracy', metrics.categorical_accuracy])

#Gerando a base de treinamento
train_datagen = ImageDataGenerator(rescale = 1./255,
									rotation_range=30,
									zoom_range=0.20,
									brightness_range=[0.6, 1.4])
#width_shift_range=0.2,
#height_shift_range=0.2
#Gerando a base de validação
validation_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_data_dir,
                                                 target_size= (img_width,img_height),
                                                 shuffle=True,
                                                 color_mode = 'grayscale',
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')

validation_set = validation_datagen.flow_from_directory(validation_data_dir,
														target_size = (img_width,img_height),
														shuffle=False,
														color_mode  = 'grayscale',
														batch_size  = 1,
														class_mode  = 'categorical')

print("Indices: "+str(training_set.class_indices))
#Número do conjunto de treinamento
print("====>"+str(training_set.n))

history = classifier.fit_generator(training_set,
                         steps_per_epoch = (training_set.n//batch_size) * 5,
                         epochs = epochs,
                         validation_data=validation_set,
                         validation_steps = nb_validation_samples,
                         verbose=1)

classifier.summary()
#score_validation = classifier.evaluate_generator(validation_set, 140)
score_validation = classifier.evaluate_generator(generator=validation_set,steps=epochs,verbose=1)
score = classifier.predict_generator(validation_set, 140)

#print(str(score))
print("Accuracy = ","%.2f" % (score_validation[1]*100),"%")
print("Loss = ","%.2f" % (score_validation[0]*100),"%")

model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    #Carregar o json e classificar com a imagem total
classifier.save_weights("model_weights.h5")
classifier.save("model.h5")


acc = history.history['acc']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
#plt.style.use("ggplot")
plt.plot(epochs, acc, 'bo', label='Training acc',color="blue")
plt.plot(epochs, val_acc, 'b', label='Validation acc',color="green")
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#Proximo passo, pegar a imagem original, pecorrer em dois for, para cortar e pintar o centro, 
#Ao pecorrer na escala de 50x50 para achar o positivo e negativo
#para isso usar a lib PIL

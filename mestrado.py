import numpy as np
import keras
#import matplotlib.pyplot as plt
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import os
import glob
from PIL import Image

def get_immediate_subdirectories(data_directory):
    return [name for name in os.listdir(data_directory)
            if os.path.isdir(os.path.join(data_directory, name))]


train_data_dir = 'data/train' 
validation_data_dir = 'data/validation'
nb_train_samples = 1000
nb_validation_samples = 36
epochs = 10
batch_size = 32  #Numero de classes
# dimensions of our images of validation.
img_width, img_height = 50, 50

adam = Adam(lr=0.001)#Adam(lr=0.001)
rmsprop = RMSprop() #RMSprop(lr=0.0005)

#Iniciando a construcao da rede convolucional
#Criando o modelo
classifier = Sequential()
# 1st Layer
classifier.add(Conv2D(40, (5, 5), input_shape = (img_width, img_height, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(1,1)))
#classifier.add(BatchNormalization())
#classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
#O dropout consiste em configurar aleatoriamente uma taxa de fracao de unidades de entrada para 0
#em cada atualizacao durante o tempo de treinamento, o que ajuda a prevenir o overfitting.
classifier.add(Dropout(0.2))
classifier.add(Flatten())
#Adicionando as camadas
#1 fully connected layers
classifier.add(Dense(units = 80, activation = 'relu'))
classifier.add(Dropout(0.2))
#2 fully connected layers
classifier.add(Dense(units = 100, activation = 'relu'))
classifier.add(Dropout(0.2))
#3 fully connected layers
classifier.add(Dense(units = 150, activation = 'relu'))
classifier.add(Dropout(0.2))
#4 fully connected layers
classifier.add(Dense(units = 5, activation = 'softmax'))
classifier.compile(optimizer = rmsprop, loss = 'categorical_crossentropy', metrics = ['accuracy', metrics.categorical_accuracy])

#Gerando a base de treinamento
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   width_shift_range=0.02,
                                   height_shift_range=0.02,
                                    rotation_range=10,
                                    zoom_range=0.10,
                                    brightness_range=[0.7, 1.3])
#width_shift_range=0.2,
#height_shift_range=0.2
#Gerando a base de validacao
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
validation_set.reset()
print("Indices: "+str(training_set.class_indices))
#Numero do conjunto de treinamento
print("====>"+str(training_set.n))

history = classifier.fit_generator(training_set,
                         steps_per_epoch = (training_set.samples//batch_size) * 30,
                         epochs = epochs,
                         validation_data=validation_set,
                         validation_steps = validation_set.samples,
                         verbose=1)

classifier.summary()
#score_validation = classifier.evaluate_generator(validation_set, 140)
score_validation = classifier.evaluate_generator(generator=validation_set,steps=validation_set.n,verbose=1)
score = classifier.predict_generator(validation_set, 140)

#print(str(score))
#print("Accuracy = ","%.2f" % (score_validation[1]*100),"%")
#print("Loss = ","%.2f" % (score_validation[0]*100),"%")


#inicializacao
mylist = []
mylistY = []

#pegando subpastas com cada classe
dir_list = get_immediate_subdirectories(validation_data_dir)
dir_list.sort()

curr_class=-1
target_names = ['', '', '', '', '']

for data_dir in dir_list: #para cada pasta de classe
    curr_class=curr_class+1
    target_names[curr_class]=data_dir
    
    #pega as imagens da pasta
    image_list = [os.path.basename(x) for x in glob.glob(os.path.join(validation_data_dir, data_dir) + '/*.jpg')]
    for img in image_list: #para cada imagem da passa da classe
        img_pil_1 = Image.open(validation_data_dir + '/' + data_dir + '/' + img).convert('L') #convertendo para greyscale
        img_pil_150 = img_pil_1.resize((img_width,img_height)) #resize
        img_np_150 = (np.asarray(img_pil_150).astype('float32'))/255 #transformando em array de floats de [0,1]
        img_np_150_rsh = img_np_150.reshape(1,img_width,img_height,1) #reshape
        pred=classifier.predict(img_np_150_rsh)[0] #predict
        mylist.append(pred) #coloca o predict na lista de predicoes
        mylistY.append(curr_class) #coloca o valor verdadeiro na lista de valores verdadeiros

y_pred = np.argmax(mylist, axis=1)

print('Accuracy')
print(accuracy_score(mylistY,y_pred))

print('Confusion Matrix')
print(confusion_matrix(mylistY,y_pred))

#exit()


model_json = classifier.to_json()
with open("Model/model.json", "w") as json_file:
    json_file.write(model_json)
    #Carregar o json e classificar com a imagem total
classifier.save_weights("Model/model_weights.h5")
classifier.save("Model/model.h5")

exit()

acc = history.history['acc']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
#plt.style.use("ggplot")
#plt.plot(epochs, acc, 'bo', label='Training acc',color="blue")
#plt.plot(epochs, val_acc, 'b', label='Validation acc',color="green")
#plt.title('Training and validation accuracy')
#plt.legend()
#plt.figure()
#plt.plot(epochs, loss, 'bo', label='Training loss')
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
#plt.title('Training and validation loss')
#plt.legend()
#plt.show()

#Proximo passo, pegar a imagem original, pecorrer em dois for, para cortar e pintar o centro, 
#Ao pecorrer na escala de 50x50 para achar o positivo e negativo
#para isso usar a lib PIL
import numpy as np
from skimage.io import imread_collection
import glob
import os
import random
import tensorflow as tf


def read_images(path):
    print('Loading, reading images')
    classes = glob.glob(path+"*")
    im_files = []
    size_classes = []
    count = 0;
    for i in classes:
        name_images_per_class = glob.glob(i+'*')
        im_files = im_files + name_images_per_class
        size_classes.append(len(name_images_per_class))
        labels = np.zeros((len(im_files),len(classes)))

    ant = 0
    for id_i, i in enumerate(size_classes):
        labels[ant: ant+i,id_i]=1
        ant=i
    collection = imread_collection(im_files)

    data =[]
    for id_i, i in enumerate(collection):
        data.append((i.reshape(3,-1)))
        return np.asarray(data), np.asarray(labels)


path = "C:/Users/Miguel Lima/Documents/Ensino/Mestrado/Linha de Pesquisa/IA/Imagens/Casas/CasasAvulsas/"
#for nome in os.listdir(path):
#    print('>>>>'+nome)
data, labels = read_images(path)
batch_size = 16
epochs = 100
percent = 0.9

data_size=len(data)
idx = np.arange(data_size)
random.shuffle(idx)
data = data[idx]
labels = labels[idx]

#Formando o c o n j u n t o de t r e i n a m e n t o com a p o r c e nt a g em de ima ge n s
#especificado na varivel percent .
train = (data[0:np.int(data_size * percent) ,: ,:] ,labels[0:np.int(data_size * percent) ,:])

test = (data[np.int(data_size*(1-percent)):,:,:], labels[np.int(data_size*(1-percent)):,:])

train_size= len(train[0])
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for n in range(epochs):
    print("###"+str(n))
    for i in range(int(np.ceil(train_size/ batch_size))):
        print('Informações: '+str(i)+' => '+str(train_size)+ ' >>'+str(batch_size))
        if(i*batch_size+batch_size <= train_size):
            batch = (train[0][i*batch_size:i*batch_size+batch_size],train[1][i*batch_size:i*batch_size+batch_size])
        else:
            batch = (train[0][i*batch_size:],
            train[1][i*batch_size:])

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if(n%5 == 0 ):
            print("Epoca %d, acuracia do treinamento = %g"%(n, train_accuracy))
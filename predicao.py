from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import cv2
from skimage import transform
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
import numpy as np
from PIL import Image as pli
from PIL import ImageChops
import os
import time

#Metodo alternativo em numpy para carregar a iamgem e transformar em grayscale
def load(filename):
    np_image = filename.convert('L')
    np_image = np.array(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (50,50, 1))
    np_image = np.expand_dims(np_image, axis=0)
    #  np_image = preprocess_input(np_image)
    return np_image

#Metodo para carregar imagem a partir da biblioteca opencv
#junto com o codigo de pecorrer pixel a pixel
def load_image_cv(path):
    img_read = cv2.imread(path)
    img_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
    largura, altura = img_read.shape
    file_name_save = save_path+"/imgpixel"
    print("Informacoes: "+str(img_read.shape))
    print("transformando em grayscale...")
    for x in range(i,largura):
        for y in range(j,altura):
            print("Pixel: "+str(x)+" - "+str(y))
            nova_imagem = img_read[x:x+altura, y:y+largura]
            if y == 1000:
                cv2.imwrite(file_name_save+"{1}-{0}".format(y,x)+".jpeg",nova_imagem)



#Carregando o modelo em diferentes formatos (Json  e h5)
path_json_file_model = 'model/model.json'
path_h5 = 'model/model.h5'
path_h5_weight = 'model/model_weights.h5'

#Carregando os modelos definidos
loaded_model = load_model(path_h5)
loaded_model.load_weights(path_h5_weight)

#Marcacoes
# As cores sao defindas para cada classe, verde para classe L1, amarela L2, laranja L3 evermelha classe L4
verde = (0,255,0)
amarelo = (53,225,255)
laranja = (0,128,255)
vermelho =  (0, 0, 255)
azul =  (255, 0, 5)


#As imagens que foram utilizadas no treinamento e validacao foram de 50x50
pattern_img_size = 50,50
#https://towardsdatascience.com/from-raw-images-to-real-time-predictions-with-deep-learning-ddbbda1be0e4
#Caminho onde estao e para onde vao os arquivos para experimento
base_path = "data/test/"
save_path = "data/test/save5/"
path_img = "data/test/Casa141.jpg"
image_process = None
#Metodo para reduzir a imagem 
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

## a ideia e pegar a imagem original e pecorrer pixel a pixel (50x50), fazer o crop e a partir desta imagem
#  fazer o predict para saber se e uma das classes ou nao
## Quando for, tem que ser marcado com uma cor mais clara e passar para o proximo
# no final e retornado a imagem com a marcacao.
#def threshold_slow(T, image):
def threshold_slow(image):
    #Transformar a imagem em array de bytes
    image_process = np.array(image)
    #Pasta onde fica alojados os arquivos cortados
    path_crop_save = save_path+"/crop/"
    print("Carregando imagem....")
    #Nome padrao para os arquivos
    file_name_save = "/imgpixel"
    text = ""
    #Obtendo a dimenssao da imagem escolhida
    largura, altura = image.size
    #Fazendo o calculo da proporcao
    proporcao = largura/altura
    print("Largura: "+str(largura))
    print("Altura: "+str(altura))
    print("Proporcao: "+str(proporcao))
    #Obtendo o pixel da imagem
    pixels = list(image.getdata())
    numl1=numl2=numl3=numl4=numl5=0
    #print("Pixels: "+str(pixels))
    #Loop para percorrer a imagem
    for y in range(0, altura-50):
        print('linha',y)
        for x in range(0, largura-50):
            #Inicialmente e realizado o corte na dimensao de 50x50
            #Crop (Esquerda, cima, direita, baixo)
            new_image = image.crop((x,y,x+50,y+50)).copy()
            #A imagem cortada e transformada em array pois o metodo de predicacao so aceita iamgem em array
            crop_img_array = np.array(new_image)
                #Essa nova imagem cortada esta em 3 canais(RGB), com isso e acionado o metodo load
                #para transformar em array e em um canal (graysacle)
                
            #img_gray = load(new_image)
            img_gray = new_image.convert('L')
            #     img_gray.show()
            time.sleep(0.5)
            img_gray = (np.asarray(img_gray).reshape(1,50,50,1))/255.0
                #A ideia e marcar o pixel central, na imagem orignal,da que foi cortada para verificar a execucao
                #com isso ha o metodo threshold no opencv para binarizar a iamgem e posteriormente
                #calcular o momento da binarizacao da imagem, que isto e realizado pelo metodo moments
                # ret,thresh = cv2.threshold(img_gray,127,255,0)
                # M = cv2.moments(thresh)
            image_class = loaded_model.predict(img_gray,verbose=0)
            print(">>>>>>>"+str(image_class))
            #Atribui o valor da classe classificada
            # print(img_gray)
            #exit()
            pred = image_class.argmax(axis=-1)
            ''' Lembrando que o 0 e o numero 7 na regua, bem como
                        o 1 e o 8, 2 e o 13 e o 3 e 14, ja para finalizar
                        4 sao as negativas ou lixo
                        '''
            
            if pred == 0:
                numl1=numl1+1
                #   time.sleep(0.5)
                        #Cada corte e salvo na pasta declarada abaixo para comparar se realmente esta fazendo correto
                crop_write = ""
                crop_write = path_crop_save+"L1/"+file_name_save+"{0}-{1}".format(x,y)+".jpeg"
                cv2.imwrite(crop_write,crop_img_array)
                            #Aqui e localizado o centro da imagem, ou seja, do 50x50 e depois marca com a cor de cada camada
                            #declarada, neste caso a verde sao para identificar o numero 7 na regua.
                cX=x+25
                cY=y+25
                image_process = cv2.circle(image_process, (cX, cY), 5, verde, -1)
                cv2.imwrite(save_path+"regua.jpg",image_process)
            elif pred == 1:
                numl2=numl2+1
                #     time.sleep(0.5)
                crop_write = ""
                crop_write = path_crop_save+"L2/"+file_name_save+"{0}-{1}".format(x,y)+".jpeg"
                cv2.imwrite(crop_write,crop_img_array)
                cX=x+25
                cY=y+25
                image_process = cv2.circle(image_process, (cX, cY), 5, amarelo, -1)
                cv2.imwrite(save_path+"regua.jpg",image_process)
            elif pred == 2:
                numl3=numl3+1
                #         time.sleep(0.5)
                crop_write = ""
                crop_write = path_crop_save+"L3/"+file_name_save+"{0}-{1}".format(x,y)+".jpeg"
                cv2.imwrite(crop_write,crop_img_array)
                cX=x+25
                cY=y+25
                image_process = cv2.circle(image_process, (cX, cY), 5, laranja, -1)
                cv2.imwrite(save_path+"regua.jpg",image_process)
            elif pred == 3:
                numl4=numl4+1
                #          time.sleep(0.5)
                crop_write = ""
                crop_write = path_crop_save+"L4/"+file_name_save+"{0}-{1}".format(x,y)+".jpeg"
                cv2.imwrite(crop_write,crop_img_array)
                cX=x+25
                cY=y+25
                image_process = cv2.circle(image_process, (cX, cY), 5, vermelho, -1)
                cv2.imwrite(save_path+"regua.jpg",image_process)
            else:
                #print("Nernhum")
                numl5=numl5+1
                print(str(numl5)+"ª vez")
                #crop_write = ""
                #crop_write = path_crop_save+"L5/"+file_name_save+"{0}-{1}".format(x,y)+".jpeg"
                #cv2.imwrite(crop_write,crop_img_array)
                #cX=x+25
                #cY=y+25
                #image_process = cv2.circle(image_process, (cX, cY), 5, azul, -1)
                #cv2.imwrite(save_path+"regua.jpg",image_process)
                    
    print(numl1,numl2,numl3,numl4,numl5)

img = pli.open(path_img)
threshold_slow(img)
print("Finalizado...")
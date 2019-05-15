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

#Método alternativo em numpy para carregar a iamgem e transformar em grayscale
def load(filename):
   np_image = filename.convert('L')
   np_image = np.array(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (50,50, 1))
   np_image = np.expand_dims(np_image, axis=0)
 #  np_image = preprocess_input(np_image)
   return np_image

#Método para carregar imagem a partir da biblioteca opencv
#junto com o código de pecorrer pixel a pixel
def load_image_cv(path):
	img_read = cv2.imread(path)
	img_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
	largura, altura = img_read.shape
	file_name_save = save_path+"/imgpixel"
	print("Informações: "+str(img_read.shape))
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

#Marcações
# As cores são defindas para cada classe, verde para classe L1, amarela L2, laranja L3 evermelha classe L4
verde = (0,255,0)
amarelo = (53,225,255)
laranja = (0,128,255)
vermelho =  (0, 0, 255)

#As imagens que foram utilizadas no treinamento e validação foram de 50x50
pattern_img_size = 50,50
#https://towardsdatascience.com/from-raw-images-to-real-time-predictions-with-deep-learning-ddbbda1be0e4
#Caminho onde estão e para onde vão os arquivos para experimento
base_path = "data/test/"
save_path = "data/test/save5/"
path_img = "data/test/Casa279.jpg"
image_process = None
#Metodo para reduzir a imagem 
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

## a ideia é pegar a imagem original e pecorrer pixel a pixel (50x50), fazer o crop e a partir desta imagem
#  fazer o predict para saber se é uma das classes ou não
## Quando for, tem que ser marcado com uma cor mais clara e passar para o proximo
# no final é retornado a imagem com a marcação.
#def threshold_slow(T, image):
def threshold_slow(image):
	#Transformar a imagem em array de bytes
	image_process = np.array(image)
	#Pasta onde fica alojados os arquivos cortados
	path_crop_save = save_path+"/crop/"
	print("Carregando imagem....")
	#Nome padrão para os arquivos
	file_name_save = "/imgpixel"
	text = ""
	#Obtendo a dimenssão da imagem escolhida
	largura, altura = image.size
	#Fazendo o calculo da proporção
	proporcao = largura/altura
	print("Largura: "+str(largura))
	print("Altura: "+str(altura))
	print("Proporção: "+str(proporcao))
	#Obtendo o pixel da imagem
	pixels = list(image.getdata())
	#print("Pixels: "+str(pixels))
	#Loop para percorrer a imagem
	for y in range(0, altura-50):
		for x in range(0, largura-50):
			#Inicialmente é realizado o corte na dimensão de 50x50
			#Crop (Esquerda, cima, direita, baixo)
			new_image = image.crop((x,y,x+50,y+50)).copy()
			#A imagem cortada é transformada em array pois o método de predicação só aceita iamgem em array
			crop_img_array = np.array(new_image)
			#Essa nova imagem cortada está em 3 canais(RGB), com isso é acionado o método load 
			#para transformar em array e em um canal (graysacle)
			img_gray = load(new_image)
			#A ideia é marcar o pixel central, na imagem orignal,da que foi cortada para verificar a execução
			#com isso há o método threshold no opencv para binarizar a iamgem e posteriormente
			#calcular o momento da binarização da imagem, que isto é realizado pelo método moments
			ret,thresh = cv2.threshold(img_gray,127,255,0)
			M = cv2.moments(thresh)
			image_class = loaded_model.predict(img_gray,verbose=0)
			print(">>>>>>>"+str(image_class))
			#Atribui o valor da classe classificada
			pred = image_class.argmax(axis=-1)
			''' Lembrando que o 0 é o numero 7 na régua, bem como
				o 1 é o 8, 2 é o 13 e o 3 é 14, já para finalizar
				4 são as negativas ou lixo
			'''
			if pred == 0:
				#Cada corte é salvo na pasta declarada abaixo para comparar se realmente está fazendo correto
				crop_write = ""
				crop_write = path_crop_save+"L1/"+file_name_save+"{0}-{1}".format(x,y)+".jpeg"
				cv2.imwrite(crop_write,crop_img_array)
				#Aqui é localizado o centro da imagem, ou seja, do 50x50 e depois marca com a cor de cada camada
				#declarada, neste caso a verde são para identificar o número 7 na régua.
				if M["m00"] != 0:
					cX = int(M["m10"] / M["m00"])
					cY = int(M["m01"] / M["m00"])
				else:
					cX, cY = 0, 0
				image_process = cv2.circle(image_process, (cX, cY), 5, verde, -1)
				cv2.imwrite(save_path+"regua.jpg",image_process)
			elif pred == 1:
				crop_write = ""
				crop_write = path_crop_save+"L2/"+file_name_save+"{0}-{1}".format(x,y)+".jpeg"
				cv2.imwrite(crop_write,crop_img_array)
				if M["m00"] != 0:
					cX = int(M["m10"] / M["m00"])
					cY = int(M["m01"] / M["m00"])
				else:
					cX, cY = 0, 0
				image_process = cv2.circle(image_process, (cX, cY), 5, amarelo, -1)
				cv2.imwrite(save_path+"regua.jpg",image_process)
			elif pred == 2:
				crop_write = ""
				crop_write = path_crop_save+"L3/"+file_name_save+"{0}-{1}".format(x,y)+".jpeg"
				cv2.imwrite(crop_write,crop_img_array)
				if M["m00"] != 0:
					cX = int(M["m10"] / M["m00"])
					cY = int(M["m01"] / M["m00"])
				else:
					cX, cY = 0, 0
				image_process = cv2.circle(image_process, (cX, cY), 5, laranja, -1)
				cv2.imwrite(save_path+"regua.jpg",image_process)
			elif pred == 3:
				crop_write = ""
				crop_write = path_crop_save+"L4/"+file_name_save+"{0}-{1}".format(x,y)+".jpeg" 
				cv2.imwrite(crop_write,crop_img_array)
				if M["m00"] != 0:
					cX = int(M["m10"] / M["m00"])
					cY = int(M["m01"] / M["m00"])
				else:
					cX, cY = 0, 0
				image_process = cv2.circle(image_process, (cX, cY), 5, vermelho, -1)
				cv2.imwrite(save_path+"regua.jpg",image_process)
			else: print("Nernhum")

img = pli.open(path_img)
threshold_slow(img)
print("Finalizado...")
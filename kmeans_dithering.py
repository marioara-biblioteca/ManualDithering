from google.colab import drive
import cv2
import matplotlib.pyplot as plt 
from matplotlib import colors
%matplotlib inline
import random 
import numpy as np 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs

from mpl_toolkits.mplot3d import Axes3D
import tensorflow_datasets as tfds
import tensorflow as tf
import math


#functii ajutatoare
def next_power_of_2(x):
    return 1 if x == 0 else 2**math.floor(math.log2(x))

def show_img(img,n,i,colormap,title):
  plt.gcf().set_size_inches(45, 45)
  sp=plt.subplot(1, n, i)
  sp.set_title(title)
  plt.imshow(img, cmap=plt.get_cmap(colormap), vmin=0, vmax=np.amax(img))
  
def crop_image(img):
  h,w,_ = img.shape #height=im.shape[0] width=img.shape[1]
  wPow=2**math.floor(math.log2(w))
  hPow=2**math.floor(math.log2(h))
  #cropam imaginea astfel incat sa fie matrice de n x n cu n  putere a lui 2
  if(wPow < hPow):
    n=wPow
  else:
    n=hPow
  cropImg=img[(h-n)//2:h-(h-n)//2,(w-n)//2:w-(w-n)//2]
  return n,cropImg

#0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue
def rgb_to_gray(img):
  gray = np.zeros([img.shape[0], img.shape[1]])
  for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
      pixel=img[i][j][0] * 0.299  + img[i][j][1] * 0.587 + img[i][j][2] * 0.114 
      gray[i][j]=pixel
  return gray

def kmeans_color_reduction(img):
  n_colors = 12
  w,h,_ = lenna.shape
  lennaR = np.reshape(lenna,(w*h,3))
  kmeans = KMeans(init="k-means++",n_clusters=n_colors, n_init=10).fit(lennaR) #n_init = number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
  kmeans_labels = kmeans.labels_
  kmeans_cluster_centers = np.uint8(kmeans.cluster_centers_)
  #centroizii sunt paleta de culori
  reducedLenna = np.asanyarray([kmeans_cluster_centers[kmeans_labels[i]] for i in range(w*h)]) # pt fiecare pixel pun una din culorile principale  
  lennaFinal = np.reshape(reducedLenna, lenna.shape)
  return lennaFinal

#fucntie care imi genereaza matricea pentru dithering ordonat, normata 
def get_dithering_matrix(n):
  d = np.array([[0]]) #M0 = [0]
  for i in range(0,n):
    #la iteratia i am de completat 4 matrice de 2^i x 2^i
    power=pow(2,i)
    new_d = np.block([[d, np.zeros(d.shape)], [d,d]])
    #4 blocuri for pentru fiecar portiune din matrice
    #unde power e dimensiunea matricei dinainte
    #am de completat 4 matrice de power x power 
    for j in range(0, power):
      for k in range(0, power):
        new_d[j][k]=d[j][k] * 4
    for j in range(power, power*2):
      for k in range(0, power):
        new_d[j][k]=d[j-power][k] * 4 + 3
    for j in range(0, power):
      for k in range(power, 2*power):
        new_d[j][k]=d[j][k-power] * 4 + 2
    for j in range(power, 2*power):
      for k in range(power, 2*power):
        new_d[j][k]=d[j-power][k-power] * 4 + 1
    d = new_d 
  return d / np.amax(d)

def dither(d,g,n):
  o=np.zeros(d.shape)
  for x in range(1, n):
    for y in range(1, n):
      i = (x - 1) % n + 1
      j= (y - 1) % n + 1
      if g[x][y] > d[i][j]:
        o[x][y]=1
      else:
        o[x][y]=0
  return o



img = cv2.imread('/gdrive/My Drive/multimedia/img.jpeg',cv2.IMREAD_UNCHANGED)
lenna = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

lennaCopy=lenna.copy()
n,lenna=crop_image(lennaCopy) # lenna ramane in lennaCopy si lucram cu lenna
show_img(lenna,4,1,'viridis','Cropped image')

reducedLenna=kmeans_color_reduction(lenna)
show_img(reducedLenna,4,2,'viridis','Image after KMeans Quantization')
#trebuie sa transformam imaginea in 2d => adica o aducem in formatul de tonuri de gri
flattenedLenna = cv2.cvtColor(reducedLenna, cv2.COLOR_BGR2GRAY)
grayLenna=rgb_to_gray(reducedLenna);
show_img(grayLenna,4,3,'gray','Image in B/W (for shape N x N)')


d=get_dithering_matrix(int(math.log2(n)))
g=flattenedLenna / np.amax(flattenedLenna)
o=dither(d,g,n)

show_img(o,4,4,'gray','Image after manual ordered dithering')
plt.show()
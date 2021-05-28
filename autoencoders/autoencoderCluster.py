#!/usr/bin/env python
# coding: utf-8

# # Intro to Autoencoders

# ## Import TensorFlow and other libraries

# In[3]:


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


# ## Load the dataset
# To start, you will train the basic autoencoder using the Fashon MNIST dataset. Each image in this dataset is 28x28 pixels. 

# In[31]:


PATH = "./mergedConstellationsStars2Outline_Large/"
checkpoint_path = "./training_ckpt/cp-{epoch:04d}.ckpt"
#imageSaveDir = "./run1"
#if not os.path.exists(imageSaveDir):
#    os.makedirs(imageSaveDir)


# In[5]:


BUFFER_SIZE = 200
BATCH_SIZE = batch_size = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
EPOCHS = 50


# In[6]:


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


# In[7]:


inp, re = load(PATH+'train/object_0_0.jpg')
# casting to int for matplotlib to show the image
plt.figure()
plt.imshow(inp/255.0)
plt.figure()
plt.imshow(re/255.0)


# In[8]:


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                                                            
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)   
    return input_image, real_image


# In[9]:


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


# In[10]:


# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
    input_image = input_image/255.0#(input_image / 127.5) - 1
    real_image = real_image/255.0#(real_image / 127.5) - 1

    return input_image, real_image


# In[11]:


@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


# In[12]:


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


# In[13]:


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


# ## Input Pipeline

# In[14]:


train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
train_dataset = train_dataset.map(load_image_train)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(1)


# In[15]:


test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(1)


# ## Convert to numpy

# In[16]:


train_np = np.stack(list(train_dataset))
test_np = np.stack(list(test_dataset))

rgb_weights = [0.2989, 0.5870, 0.1140] #To grayscale
train_np = np.dot(train_np[...,:3], rgb_weights)
test_np = np.dot(test_np[...,:3], rgb_weights)

# In[17]:

x_train_noisy = np.squeeze(train_np[:,0,:,:,:])
x_train_noisy = x_train_noisy[:,:,:,np.newaxis]
x_train = np.squeeze(train_np[:,1,:,:,:])
x_train = x_train[:,:,:,np.newaxis]


# In[18]:

x_test_noisy = np.squeeze(test_np[:,0,:,:,:])
x_test_noisy = x_test_noisy[:,:,:,np.newaxis]
x_test = np.squeeze(test_np[:,1,:,:,:])
x_test = x_test[:,:,:,np.newaxis]

# ### Define a convolutional autoencoder

# In this example, you will train a convolutional autoencoder using  [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) layers in the `encoder`, and [Conv2DTranspose](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose) layers in the `decoder`.

# In[25]:


#Arhitecture from https://learnopencv.com/variational-autoencoder-in-tensorflow/
class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(256, 256, 1,)),
      layers.Conv2D(32, kernel_size=3, padding='same', strides=2),
      layers.BatchNormalization(name='bn_1'),
      layers.LeakyReLU(name='lrelu_1'), #Block1
      layers.Conv2D(64, kernel_size=3, activation='relu', padding='same', strides=2),
      layers.BatchNormalization(name='bn_2'),
      layers.LeakyReLU(name='lrelu_2'), #Block2
      layers.Conv2D(64, 3, 2, padding='same', name='conv_3'),
      layers.BatchNormalization(name='bn_3'),
      layers.LeakyReLU(name='lrelu_3'), #Block3
      layers.Conv2D(64, 3, 2, padding='same', name='conv_4'),
      layers.BatchNormalization(name='bn_4'),
      layers.LeakyReLU(name='lrelu_4'), #Block4
      layers.Conv2D(64, 3, 2, padding='same', name='conv_5'),
      layers.BatchNormalization(name='bn_5'),
      layers.LeakyReLU(name='lrelu_5'), #Block5
      layers.Flatten(),
      layers.Dense(1024, name='mean')])

    self.decoder = tf.keras.Sequential([
      layers.InputLayer(input_shape=(1024,)),
      layers.Dense(4096, name='dense_1'),
      layers.Reshape((8,8,64), name='Reshape'),
      layers.Conv2DTranspose(64, 3, strides= 2, padding='same',name='conv_transpose_1'),
      layers.BatchNormalization(name='bn_1'),
      layers.LeakyReLU(name='lrelu_1'),
      layers.Conv2DTranspose(64, 3, strides= 2, padding='same', name='conv_transpose_2'),
      layers.BatchNormalization(name='bn_2'),
      layers.LeakyReLU(name='lrelu_2'),
      layers.Conv2DTranspose(64, 3, 2, padding='same', name='conv_transpose_3'),
      layers.BatchNormalization(name='bn_3'),
      layers.LeakyReLU(name='lrelu_3'),
      layers.Conv2DTranspose(32, 3, 2, padding='same', name='conv_transpose_4'),
      layers.BatchNormalization(name='bn_4'),
      layers.LeakyReLU(name='lrelu_4'),
      layers.Conv2DTranspose(1, 3, 2,padding='same', activation='sigmoid', name='conv_transpose_5')])

  def call(self, x):
    print(x.shape)
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


# In[26]:


autoencoder = Denoise()


# In[27]:


# Include the epoch in the file name (uses `str.format`)
checkpoint_dir = os.path.dirname(checkpoint_path)


# Create a callback that saves the model's weights every 10 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=False,
    save_freq=len(x_train)*50)


# In[28]:


autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())


# In[29]:


autoencoder.save_weights(checkpoint_path.format(epoch=0))


# In[30]:


autoencoder.fit(x_train_noisy, x_train,
                epochs=EPOCHS,
                shuffle=True,
                callbacks=[cp_callback],
                validation_data=(x_test_noisy, x_test))


# In[32]:


#autoencoder.encoder.summary()


# In[33]:


#autoencoder.decoder.summary()

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

def generate_images_test(x_testImg, noise_testImg, x_outImg, saveFolder, index):
    
    inp = Image.fromarray(np.squeeze((noise_testImg)*255).astype(np.uint8), "L")
    inp.save(f"{saveFolder}/input_{index}.jpg")
    
    tar = Image.fromarray((np.squeeze(x_testImg)*255).astype(np.uint8), "L")
    tar.save(f"{saveFolder}/target_{index}.jpg")
    
    pred = Image.fromarray((np.squeeze(x_outImg)*255).astype(np.uint8), "L")
    pred.save(f"{saveFolder}/predictions_{index}.jpg")
    
saveDir = PATH+"predictions_ae"
for i in range(100):
    generate_images_test(x_test[i], x_test_noisy[i], decoded_imgs[i], saveDir, i)
    



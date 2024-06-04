#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
!pip install opendatasets
import opendatasets as od
from PIL import Image
from skimage import io
import pathlib

import cv2
import seaborn as sns
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator




#import dataset using kaggle login 
url = "https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images"
od.download(url)






#--data preprocessing
#for loop to get the path of every image in cifake folder
name_of_folder = "cifake-real-and-ai-generated-synthetic-images"
focus = pathlib.Path("/content/" + name_of_folder)
all_files_list = list(focus.rglob("*"))

#use information from path string to sort file paths
FAKE = []
REAL = []
train_fake = []
train_real = []
test_fake = []
test_real = []
for i in all_files_list:
  currentFile = str(i)
  if ".jpg" in currentFile: #if its an image


    if "train" in currentFile: #if its train

      if "REAL" in currentFile:
        train_real.append(currentFile)

      if "FAKE" in currentFile:
        train_fake.append(currentFile)


    elif "test" in currentFile: #if its test

      if "REAL" in currentFile:
        test_real.append(currentFile)

      if "FAKE" in currentFile:
        test_fake.append(currentFile)

FAKE = train_fake+test_fake
REAL = test_real+test_fake






#--color intensity visualizations
def colorValue (img):
  redSum = 0
  blueSum = 0
  greenSum = 0
  #red sum
  for row_index in range(img.shape[0]) :
    for column_index in range(img.shape[1]):
      redSum+=(img[row_index][column_index][0])
      blueSum+=(img[row_index][column_index][2])
      greenSum+=(img[row_index][column_index][1])

  numPixels = (img.shape[0])*(img.shape[1])
  return [(redSum/numPixels), (greenSum/numPixels), (blueSum/numPixels)]

#obtain all averages of rgb values in a list
redValues = []
blueValues = []
greenValues = []
redValuesReal = []
blueValuesReal = []
greenValuesReal = []

for imagePath in FAKE:
  img = io.imread(imagePath)
  colorMeans = colorValue(img)
  redValues.append(colorMeans[0])
  blueValues.append(colorMeans[2])
  greenValues.append(colorMeans[1])

for imagePath in REAL:
  img = io.imread(imagePath)
  colorMeans1 = colorValue(img)
  redValuesReal.append(colorMeans1[0])
  blueValuesReal.append(colorMeans1[2])
  greenValuesReal.append(colorMeans1[1])

#histogram for red values (ai generated and real images)
plt.hist(redValues)
plt.xlabel("Average Red Color Channel Intensity")
plt.ylabel("Number of Images")
plt.title("Average Red Color Channel Intensity for AI Generated Images vs Number of Images")

plt.hist(redValuesReal)
plt.xlabel("Average Red Color Channel Intensity")
plt.ylabel("Number of Images")
plt.title("Average Red Color Channel Intensity for Human Produced Images vs Number of Images")

#boxplot for blue values (ai generated and real images)
plt.boxplot(blueValuesReal)
plt.xlabel("Average Blue Color Channel Intensity")
plt.ylabel("Number of Images")
plt.title("Average Blue Color Channel Intensity for Human Produced Images vs Number of Images")

plt.boxplot(blueValues)
plt.xlabel("Average Blue Color Channel Intensity")
plt.ylabel("Number of Images")
plt.title("Average Blue Color Channel Intensity for AI Generated Images vs Number of Images")

#histogram and boxplot for green values (ai generated and real images)
plt.hist(greenValues)
plt.xlabel("Average Green Color Channel Intensity")
plt.ylabel("Number of Images")
plt.title("Average Green Color Channel Intensity for AI Generated Images vs Number of Images")

plt.boxplot(greenValuesReal)
plt.xlabel("Average Green Color Channel Intensity")
plt.ylabel("Number of Images")
plt.title("Average Green Color Channel Intensity for Human Produced Images vs Number of Images")








#average brightness visualizations
from PIL import Image
import numpy as np

def brightnessFunc (img):
  brightnessValues = []

  img_np = np.array(img)
  gray_img = img.convert('L')
  gray_img_np = np.array(gray_img)

  for row_index in range (gray_img_np.shape[0]):
    for column_index in range (gray_img_np.shape[1]):
      brightnessValues.append(gray_img_np[row_index][column_index])

  totalBrightness = sum(brightnessValues)

  numPixels = (gray_img_np.shape[0])*(gray_img_np.shape[1])
  return totalBrightness/numPixels

avg_brightness_values_fake = []
avg_brightness_values_real = []

for imagePath in FAKE:
  img = Image.open(imagePath)
  avg_brightness_values_fake.append(brightnessFunc(img))

for imagePath in REAL:
  img = Image.open(imagePath)
  avg_brightness_values_real.append(brightnessFunc(img))


#graph ai generated images on a histrogram 
plt.hist(avg_brightness_values_fake, edgecolor = "black", linewidth = 0.9)
plt.xlabel("Average Color Intensity")
plt.ylabel("Number of Images")
plt.title("Average Color Intensity for AI Generated Images vs Number of Images")

max_y = plt.gca().get_ylim()[1]
plt.yticks(np.arange(0, max_y + 2500, 2500))

# Add horizontal grid lines at y-tick positions
plt.grid(axis='y', linestyle='--', linewidth=0.5)

bin_edges = np.histogram_bin_edges(avg_brightness_values_fake, bins=10)
for edge in bin_edges:
    plt.axvline(edge, color='black', linestyle='--', linewidth=0.5)

plt.tight_layout()

#graph real images on a histogram
plt.hist(avg_brightness_values_real, edgecolor = "black", linewidth = 0.9)
plt.xlabel("Average Color Intensity")
plt.ylabel("Number of Images")
plt.title("Average Color Intensity for Human Produced Images vs Number of Images")

#add lines to separated data
bin_edges = np.histogram_bin_edges(avg_brightness_values_real, bins=10)
for edge in bin_edges:
    plt.axvline(edge, color='black', linestyle='--', linewidth=0.5)

max_y = plt.gca().get_ylim()[1]
plt.yticks(np.arange(0, max_y + 2500, 2500))

# Add horizontal grid lines at y-tick positions
plt.grid(axis='y', linestyle='--', linewidth=0.5)

plt.tight_layout()











#gradient magnitude visualizations
from scipy.ndimage import sobel

def average_gradient(image_path):
    img_gray = img.convert('L')
    img_np = np.array(img_gray)

    # Compute the gradient magnitude for the entire image
    gradient_magnitude = sobel(img_np)

    # Compute the total gradient magnitude
    total_gradient = np.sum(gradient_magnitude)

    # Compute the number of pixels
    num_pixels = img_np.shape[0] * img_np.shape[1]

    # Compute the average gradient magnitude
    avg_gradient_mag = total_gradient / num_pixels

    return avg_gradient_mag

total_gradient_values_fake = []
total_gradient_values_real = []

for imagePath in FAKE:
  img = Image.open(imagePath)
  total_gradient_values_fake.append(average_gradient(img))

for imagePath in REAL:
  img = Image.open(imagePath)
  total_gradient_values_real.append(average_gradient(img))

#histogram plot for fake images 
plt.hist(total_gradient_values_fake)
plt.xlabel("Gradient Magnitude")
plt.ylabel("Number of Images")
plt.title("Gradient Magnitude for AI Generated Images vs Number of Images")

#histogram plot for real images 
plt.hist(total_gradient_values_real)
plt.xlabel("Gradient Magnitude")
plt.ylabel("Number of Images")
plt.title("Gradient Magnitude for Human Produced Images vs Number of Images")

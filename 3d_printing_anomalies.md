---
layout: default
title: 3d Printing Anomalies
permalink: /3d_printing_anomalies
description: "Using autoencoders to find issues in 3d printing images"
---
# 3D Printing Anomalies
## Background

I recently purchased a 3D printer and while creating or finding the perfect design can take a while, the hardest part of the process is the actual printing. If the settings on the printer are not perfect or the printer is nudged slightly during the process, you can end up with several different issues with your print. This can also vary depending on the spool, even if it is the exact type you have used before, small changes in quality can ruin your print. If you leave your print running overnight, which is necessary due to the length of some prints, you could wake up the next morning finding the printer stopped extruding due to a kink in the line, wasting energy due to keeping the nozzle hot, or worse, stringy filament everywhere because it was bumped or wouldn't stick wasting both energy and filament. One way to catch this issue early if you were consistently printing the same model would be to use an autoencoder to recreate images and find anomalies. Using image data from Kaggle showing normal prints and prints with issues, that is what this project attempts to achieve.

## The Data

For this project, I used data from the popular machine learning website `www.kaggle.com`. The specific data source I used can be found at `https://www.kaggle.com/datasets/justin900429/3d-printer-defected-dataset`. The data consists of images split into two folders. The first folder contains 798 images of prints in progress with no known defects. The second folder contains 758 images of prints with defects split into four categories:
- Bed not sticking
- Broken leg
- No bottom
- No support

## Issues covered
### Bed not sticking
In this issue, the initial layer of filament has not stuck to the bed of the printer. This is usually shown by stringy filament being dragged around the bed or bunching up around the nozzle, potentially causing a clog. The issue can be fixed in two ways, either by adjusting the heat of the bed or applying glue to the bed before the print. There are 94 images in this set. An example image of this issue is shown below.

![bed not sticking example](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/aea2615c-be76-490a-b733-19fa1e4c8761)

### Broken leg
In This issue, large blobs of filament are appearing along with regular-sized extruded filament. This may indicate a problem with the quality of the filament being used or the temperature of the nozzle. There are 214 images in this set.

![leg broken example](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/0e941988-9997-4ca5-81c9-90711acc626e)


### No bottom
In this issue, the first few layers may have printed fine but then become unstuck leading to future layers becoming stringy. This may have been caused by the printer being bumped, the nozzle extruding too close to the previous layer or the bed not being sticky enough. Potential solutions include adjusting the layer height of the print, adjusting the temperature of the bed, or applying glue before printing. There are 96 images in this set.

![no bottom example](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/b2009c73-303c-4551-b7a6-71f2e0a56b90)

### No support
In this issue, there is nothing to support the initial layer of the print. This results in the nozzle extruding a sting of filament which has nothing to stick to. The issue can be solved by levelling the bed and adjusting the z-offset of the printer. When the height is set to 0, the nozzle should be the width of an A4 sheet of paper from the bed, a common test is to slide a sheet of paper between the nozzle and the bed at multiple positions and adjust the bed height and z offset until it starts vibrating. There are 355 images in this set.

![no support example](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/fdc3f9e8-fe8b-4b7f-adf6-995dbb5974f0)


## Data extraction
As mentioned above, the images are split into two folders the defected images are stored in the `defected` folder and the regular images in the `no_defected` folder. These are stored under the parent folder `archive`. Going through each of these folders, I used the cv2 python package to extract the images and store them as a list of numpy arrays, I also resized the images to ensure that there was a constant image size being fed to the network. Along with the images, I stored the labels as a list of integers based on which folder the image was in. I then converted both lists to numpy arrays to be fed into the training script. You can see the code for this section below.

```
def extract_images() -> Tuple[np.ndarray, np.ndarray]:
    data_directory = os.path.join(os.getcwd(), "archive")
    subfolders = [folder for folder in os.listdir(data_directory)]
    images: List[np.ndarray] = []
    labels: List[int] = []
    for subfolder in subfolders:
        for image in os.listdir(os.path.join(data_directory,subfolder)):
            img = cv2.imread(os.path.join(data_directory, subfolder, image))
            img = cv2.resize(img, (400,400))
            images.append(np.asarray(img))
            if subfolder.startswith('no_'):
                labels.append(1)
            else:
                labels.append(0)
    images = np.asarray(images)
    labels = np.asarray(labels)
    return images, labels
```

## Training
There are four main parts to the training file I built:
- Building the dataset
- Defining the autoencoder
- Running the autoencoder
- Saving the outputs

### Building the dataset
Building the dataset involves taking the output from the data extraction step and producing a dataset suitable for the autoencoder. First I identified the indexes for each label type, then shuffled them so each image has an equal chance of being used in the model. I then used the contamination percentage to work out how many anomaly images I wanted to use and assigned that many indexes to a new variable. The indexes were then used to create new numpy arrays of the valid and anomaly images. An array of labels was created to identify the anomaly images. The valid and anomaly arrays were then stacked together to form a single image array. The images and label arrays were then shuffled using an array of indexes to ensure they were shuffled together. The array of labels was then written out to a file to be used by the review file and the function returned the array of images. You can see the code for this below.

```
def build_dataset(data: np.ndarray, labels: np.ndarray, valid_label: int =1, anomaly_label: int =0, contamination: float = 0.01, seed: int = 77) -> np.ndarray:
    valid_indexes = np.where(labels == valid_label) [0]
    anomaly_indexes = np.where(labels == anomaly_label) [0]

    random.seed(seed)
    random.shuffle(valid_indexes)
    random.shuffle(anomaly_indexes)

    anomalies = int(len(valid_indexes) * contamination)
    anomaly_indexes = anomaly_indexes[:anomalies]

    valid_images = data[valid_indexes]
    anomaly_images = data[anomaly_indexes]

    labels_list = []
    for image in valid_images:
        labels_list.append(valid_label)
    for image in anomaly_images:
        labels_list.append(anomaly_label)
    labels_array = np.asarray(labels_list)
    images = np.vstack((valid_images, anomaly_images))
    np.random.seed(seed)
    randomise = np.arange(len(images))
    np.random.shuffle(randomise)
    images = images[randomise]
    labels_array = labels_array[randomise]
    with open(os.path.join('labels',f"{date_time}.txt"),'w+') as f:
        for label in range(len(labels_array)):
            if labels_array[label] == anomaly_label:
                f.write(f"{label}\n")
    return images
```

### Defining the autoencoder
Before I built the autoencoder, I needed to define factors like the number of epochs, the learning rate and the batch size. I also had to initialise the autoencoder with the input size, nodes sizes window and step size for tiling and the latent dimension size (As these variables are usually trial and error, I wanted to make the autoencoder builder as modular as possible). We can then compile and run the autoencoder as shown in the code below.

```
Epochs = 125
Init_LR = 1e-3
batch_size = 32

print("loading dataset")
data, labels = extract_images()
print("creating dataset")
images = build_dataset(data, labels, valid_label=1, anomaly_label=0, contamination=0.01)

images = images.astype("float32")/255.0
train_x, test_x = train_test_split(images, test_size=0.2, random_state=77)

print("building autoencoder")
autoencoder = Autoencoder.build(400,400,3, (16, 8), 3, 2, 16)
optimiser = Adam(learning_rate = Init_LR, decay = Init_LR/Epochs)
autoencoder.compile(loss = 'mae', optimizer = optimiser)

model = autoencoder.fit(train_x, train_x,
                        validation_data = (test_x, test_x),
                        epochs = Epochs,
                        batch_size = batch_size)
print("running predictions")
decoded = autoencoder.predict(test_x)
```

### Saving outputs
Finally, there are a number of outputs of this file that need to be saved including:
- A sample of images and their reconstruction
- The loss curve
- The images used
- The model

#### Image reconstructions
A sample of the images and their reconstructions is needed for the user to judge qualitatively the strength of the reproduction. The reproductions and the originals are fed into the function along with the number of samples. The original and sample are multiplied by 255 and converted to unsigned integers to undo the normalisation performed during preprocessing. each original and reconstructed image is then stacked horizontally and each sample is stacked vertically. The function then outputs the new array which is saved to a file using the cv2 package. You can see the code for this below.

```
def show_predictions(decoded: np.ndarray, gt: np.ndarray, samples: int=10) -> np.ndarray:
    for sample in range(0, samples):
        original = (gt[sample] * 255).astype("uint8")
        reconstructed = (decoded[sample] * 255).astype("uint8")
        output = np.hstack([original, reconstructed])
        if sample == 0:
            outputs = output
        else:
            outputs = np.vstack([outputs, output])
    return outputs
vis = show_predictions(decoded, test_x)
cv2.imwrite(os.path.join("reconstruction", f"{date_time}.png"), vis)
```
The image below shows an example of the image reconstruction.

![Image reconstruction](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/9f479516-ab8b-42df-8a38-4a0e066cbb35)


#### Loss curve
The loss curve helps the user determine which epoch the loss levels out to avoid overfitting. It includes lines representing the training data, which the model is trained on, and the test or validation data, which is used to show how the model performs on data it has not seen. For the model I ended up using, we can see that both sets of data stabilise around 125 epochs.

You can see the code to generate this graph below and the graph itself below.

```
N = np.arange(0, Epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, model.history["loss"], label = "train_loss")
plt.plot(N, model.history["val_loss"], label = "val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(os.path.join("loss",f"{date_time}.jpg"))
```

![21-02-2024-10-45-33](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/79f27f2b-5f9a-4aa4-aff5-1fb415204f59)


#### Images used
The data used to train the model is needed in the review stage. Because of this, I saved the data as a pickle to be extracted by another file.

```
print("saving data")
dataset = os.path.join("output",f"{date_time}.pickle")
f = open(dataset, 'wb')
f.write(pickle.dumps(images))
f.close()
```

#### Model
The state of the model is also needed in the review stage. As such, I used tensorflow's built-in model saver to save the model.

```
print("saving autoencoder")
model_out = os.path.join("model", f"{date_time}.model")
autoencoder.save(model_out, save_format='h5')
```

## Autoencoder
An autoencoder is a neural network that attempts to recreate the image it was initially given. This is achieved by passing the image through an encoder and then passing the output back through a decoder to produce a new image. Due to the informed guess and check nature of building a convolutional neural network model, I wanted to make sure the autoencoder was as modular as possible. As such, the function accepts parameters for the image height, width and depth(number of channels), the number of nodes in each convolutional layer, the window and stride for tiling, and the latent dimension for the dense layer of the encoder.

### Encoder
The encoder starts with an input layer based on the shape of the image. The code then cycles through each node creating a convolutional layer for each based on the tiling parameters with relu activation. After the final node, the network is flattened to be fed into a dense layer based on the latent dimensions input. This final output along with the input is then saved as the encoder model. You can see the code for this below.

```
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from typing import Tuple

class Autoencoder():
    def build(height:int, width:int, depth:int, nodes: Tuple[int], window: int, stride: int, latent_dim: int) -> Model:
        shape = (width, height, depth)
        inputs = layers.Input(shape)
        x = inputs
        for node in nodes:
            x = layers.Conv2D(node, (window,window), activation='relu', padding='same', strides=stride)(x)
        conv_shape = x.shape
        x = layers.Flatten()(x)
        x = layers.Dense(latent_dim + latent_dim)(x)
        latent = x
        encoder = Model(inputs, x, name = 'encoder')
```

### Decoder
The decoder starts with a Dense layer with the number of nodes equal to multiplying the shape of the last convolutional layer of the encoder network. This is to ensure there are enough nodes to rebuild the network. I then added a reshape layer to have the data in the right shape. The nodes are then traversed in reverse with each node adding a transposed convolutional layer with the same activation, window and stride as before. For the output, we use a convolutional layer with sigmoid activation and the number of nodes equal to the image depth. The decoder model is then saved with the encoder's outputs as its inputs and the current output as its output. With both the encoder and decoder now built, we can save the autoencoder with the initial inputs as its input and the result of running the decoder on the encoder's outputs as its outputs. This new autoencoder model is now returned by the function. You can see this second half of the function's code below.

```
        x = layers.Dense(conv_shape[1]*conv_shape[2]*conv_shape[3])(x)
        x = layers.Reshape(target_shape = conv_shape[1:])(x)
        reverse_nodes = nodes[::-1]
        for node in reverse_nodes:
            x = layers.Conv2DTranspose(node, kernel_size=window, strides=stride, activation="relu", padding="same")(x)
        x = layers.Conv2D(depth, kernel_size = (window, window), activation="sigmoid", padding="same")(x)
        outputs = x
        decoder = Model(latent, outputs, name = 'decoder')
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        return autoencoder
```

## Review Results
The file that reviews the results of the autoencoder can be split into four parts:
- Load the data/model
- Calculate errors
- Find the most anomalous images
- Share results

### Load data and model
When we ran the training file, the model, images and labels were saved to separate files. These files are now loaded to assess the results. The code for this is shown below:

```
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import random
from extract import extract_images
import cv2
import os

timestamp = "21-02-2024-10-45-33"
dataset = os.path.join("output",f"{timestamp}.pickle")
model = os.path.join("model",f"{timestamp}.model")
anomaly_file = os.path.join("labels", f"{timestamp}.txt")

autoencoder = load_model(model)
images = pickle.loads(open(dataset, "rb").read())
anomaly_indexes = []
with open(anomaly_file, 'r') as f:
    for anomaly_index in f.readlines():
        anomaly_indexes.append(int(anomaly_index))
```

### Calculate errors
The next step is to calculate the overall and by-pixel errors in each image. The overall mean squared error allows us to identify the worst reconstructed images. The error by pixel allows us to create a heatmap showing the worst-performing areas of each image. After running the images through the model's prediction function, we calculate the overall mean squared error, stored as a float and the mean squared error for each pixel, stored as a numpy array. You can see the code for this section below.

```
decoded = autoencoder.predict(images)
errors = []
pxl_errors = []

for (image, reconstruction) in zip(images, decoded):
    if len(image.shape) == len(reconstruction.shape) + 1:
        image = image.reshape(image.shape[:-1])
    mse = np.mean((image - reconstruction)**2)
    pxl_error = np.mean((image - reconstruction)**2, axis=2)
    errors.append(mse)
    pxl_errors.append(pxl_error)
```

### Finding the most anomalous images
Now that we have a list of the overall error for each image, we can use this to find the images most likely to have defects. We first use numpy to find the 98th quantile (assuming normal distribution) of the errors. This means that anything above this will be in the top 2% of errors. A new list of indexes is created along with the overall and by-pixel errors for any error above this threshold. The code for this section is shown below.

```
threshold = np.quantile(errors, 0.98)
anomalies = np.where(np.array(errors) >= threshold)[0]
anomaly_errors = [error for error in errors if error >= threshold]
anomaly_pxl_errors = [pxl_error for pxl_error, error in zip(pxl_errors,errors) if error>=threshold]
print(f"mse threshold: {threshold}")
```

### Share results
Each image on this new list is checked against its label to see if it was one of the defect images. The results show that with the threshold set to 98%, the model finds 6 of the 7 defect images. For a more qualitative assessment, the code also writes the original, reproduction, and heatmap for each image in the error range to a new file. after some preprocessing, the heatmap is generated using the `applyColormap` function of the cv2 package. The original, reconstruction and heatmap are then stacked horizontally for each image and each image is stacked vertically. This output is then written to a file using cv2. This is done using the code below.

```
anomaly_count = len(anomaly_indexes)
print(f"{len(anomalies)} anomalies found of {len(images)} images and {anomaly_count} anomalies")
sum = 0
for anomaly in anomalies:
    if anomaly in anomaly_indexes:
        sum+=1
print(f"success rate {sum} out of {anomaly_count}")

outputs = None
for error, anomaly, pxl_error in sorted(zip(anomaly_errors, anomalies, anomaly_pxl_errors), reverse=True):
    original = (images[anomaly] * 255).astype("uint8")
    reconstruction = (decoded[anomaly] * 255).astype("uint8")
    heatmap = np.expand_dims(np.round((pxl_error/np.max(pxl_error))*255), axis=-1).astype('uint8')
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    output = np.hstack([original, reconstruction,
                         heatmap_img
                         ])
    if outputs is None:
        outputs = output
    else:
        outputs = np.vstack([outputs, output])
    print(anomaly in anomaly_indexes)
cv2.imwrite(os.path.join("anomalies",f"{timestamp}.png"), outputs)
```

An example of the images alongside the heatmaps is shown below. We can see examples of stringy filament being highlighted as well as the user's hand when they move it into the shot. The wheels of the extruder also appear to raise errors in some images. Taking images less often, whenever the extruder is in a specific position may lead to fewer errors.

![21-02-2024-10-45-33](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/9f9111cb-84f0-4f38-a05b-5947e02638ba)


## [Return Home](https://sammatt87.github.io/)
---
layout: default
title: 3d Printing Anomalies
permalink: /3d_printing_anomalies
description: "Using autoencoders to find issues in 3d printing images"
---
# 3D Printing Anomalies
## Background

I recently purchased a 3D printer and while creating or finding the perfect design can take a while, the hardest part of the process is the actual printing. If the settings on the printer are not perfect or the printer is nudged slightly during the process, you can end up with several different issues with your print. This can also vary depending on the spool, even if it is the exact type you have used before, small changes in quality can ruin your print. If you leave your print running overnight, which is necessary due to the length of some prints, you could wake up the next morning finding the printer stopped extruding due to a kink in the line, wasting energy due to keeping the nozzle hot, or worse, stringy filament everywhere because it was bumped or wouldn't stick wasting both energy and filament. One way to catch this issue early if you were consistently printing the same model would be to use an autoencoder to recreate images and find anomalies. Using image data from Kaggle showing normal prints and prints with issues, that is what this project attempts to achieve.

## The Data

For this project, I used data from the popular machine learning website `www.kaggle.com`. The specific data source I used can be found at `https://www.kaggle.com/datasets/justin900429/3d-printer-defected-dataset`. The data consists of images split into two folders. The first folder contains 798 images of prints in progress with no known defects. The second folder contains 758 images of prints with defects split into four categories:
- Bed not sticking
- Broken leg
- No bottom
- No support

## Issues covered
### Bed not sticking
In this issue, the initial layer of filament has not stuck to the bed of the printer. This is usually shown by stringy filament being dragged around the bed or bunching up around the nozzle, potentially causing a clog. The issue can be fixed in two ways, either by adjusting the heat of the bed or applying glue to the bed before the print. There are 94 images in this set. An example image of this issue is shown below.

![bed not sticking example](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/aea2615c-be76-490a-b733-19fa1e4c8761)

### Broken leg
In This issue, large blobs of filament are appearing along with regular-sized extruded filament. This may indicate a problem with the quality of the filament being used or the temperature of the nozzle. There are 214 images in this set.

![leg broken example](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/0e941988-9997-4ca5-81c9-90711acc626e)


### No bottom
In this issue, the first few layers may have printed fine but then become unstuck leading to future layers becoming stringy. This may have been caused by the printer being bumped, the nozzle extruding too close to the previous layer or the bed not being sticky enough. Potential solutions include adjusting the layer height of the print, adjusting the temperature of the bed, or applying glue before printing. There are 96 images in this set.

![no bottom example](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/b2009c73-303c-4551-b7a6-71f2e0a56b90)

### No support
In this issue, there is nothing to support the initial layer of the print. This results in the nozzle extruding a sting of filament which has nothing to stick to. The issue can be solved by levelling the bed and adjusting the z-offset of the printer. When the height is set to 0, the nozzle should be the width of an A4 sheet of paper from the bed, a common test is to slide a sheet of paper between the nozzle and the bed at multiple positions and adjust the bed height and z offset until it starts vibrating. There are 355 images in this set.

![no support example](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/fdc3f9e8-fe8b-4b7f-adf6-995dbb5974f0)


## Data extraction
As mentioned above, the images are split into two folders the defected images are stored in the `defected` folder and the regular images in the `no_defected` folder. These are stored under the parent folder `archive`. Going through each of these folders, I used the cv2 python package to extract the images and store them as a list of numpy arrays, I also resized the images to ensure that there was a constant image size being fed to the network. Along with the images, I stored the labels as a list of integers based on which folder the image was in. I then converted both lists to numpy arrays to be fed into the training script. You can see the code for this section below.

```
def extract_images() -> Tuple[np.ndarray, np.ndarray]:
    data_directory = os.path.join(os.getcwd(), "archive")
    subfolders = [folder for folder in os.listdir(data_directory)]
    images: List[np.ndarray] = []
    labels: List[int] = []
    for subfolder in subfolders:
        for image in os.listdir(os.path.join(data_directory,subfolder)):
            img = cv2.imread(os.path.join(data_directory, subfolder, image))
            img = cv2.resize(img, (400,400))
            images.append(np.asarray(img))
            if subfolder.startswith('no_'):
                labels.append(1)
            else:
                labels.append(0)
    images = np.asarray(images)
    labels = np.asarray(labels)
    return images, labels
```

## Training
There are four main parts to the training file I built:
- Building the dataset
- Defining the autoencoder
- Running the autoencoder
- Saving the outputs

### Building the dataset
Building the dataset involves taking the output from the data extraction step and producing a dataset suitable for the autoencoder. First I identified the indexes for each label type, then shuffled them so each image has an equal chance of being used in the model. I then used the contamination percentage to work out how many anomaly images I wanted to use and assigned that many indexes to a new variable. The indexes were then used to create new numpy arrays of the valid and anomaly images. An array of labels was created to identify the anomaly images. The valid and anomaly arrays were then stacked together to form a single image array. The images and label arrays were then shuffled using an array of indexes to ensure they were shuffled together. The array of labels was then written out to a file to be used by the review file and the function returned the array of images. You can see the code for this below.

```
def build_dataset(data: np.ndarray, labels: np.ndarray, valid_label: int =1, anomaly_label: int =0, contamination: float = 0.01, seed: int = 77) -> np.ndarray:
    valid_indexes = np.where(labels == valid_label) [0]
    anomaly_indexes = np.where(labels == anomaly_label) [0]

    random.seed(seed)
    random.shuffle(valid_indexes)
    random.shuffle(anomaly_indexes)

    anomalies = int(len(valid_indexes) * contamination)
    anomaly_indexes = anomaly_indexes[:anomalies]

    valid_images = data[valid_indexes]
    anomaly_images = data[anomaly_indexes]

    labels_list = []
    for image in valid_images:
        labels_list.append(valid_label)
    for image in anomaly_images:
        labels_list.append(anomaly_label)
    labels_array = np.asarray(labels_list)
    images = np.vstack((valid_images, anomaly_images))
    np.random.seed(seed)
    randomise = np.arange(len(images))
    np.random.shuffle(randomise)
    images = images[randomise]
    labels_array = labels_array[randomise]
    with open(os.path.join('labels',f"{date_time}.txt"),'w+') as f:
        for label in range(len(labels_array)):
            if labels_array[label] == anomaly_label:
                f.write(f"{label}\n")
    return images
```

### Defining the autoencoder
Before I built the autoencoder, I needed to define factors like the number of epochs, the learning rate and the batch size. I also had to initialise the autoencoder with the input size, nodes sizes window and step size for tiling and the latent dimension size (As these variables are usually trial and error, I wanted to make the autoencoder builder as modular as possible). We can then compile and run the autoencoder as shown in the code below.

```
Epochs = 125
Init_LR = 1e-3
batch_size = 32

print("loading dataset")
data, labels = extract_images()
print("creating dataset")
images = build_dataset(data, labels, valid_label=1, anomaly_label=0, contamination=0.01)

images = images.astype("float32")/255.0
train_x, test_x = train_test_split(images, test_size=0.2, random_state=77)

print("building autoencoder")
autoencoder = Autoencoder.build(400,400,3, (16, 8), 3, 2, 16)
optimiser = Adam(learning_rate = Init_LR, decay = Init_LR/Epochs)
autoencoder.compile(loss = 'mae', optimizer = optimiser)

model = autoencoder.fit(train_x, train_x,
                        validation_data = (test_x, test_x),
                        epochs = Epochs,
                        batch_size = batch_size)
print("running predictions")
decoded = autoencoder.predict(test_x)
```

### Saving outputs
Finally, there are a number of outputs of this file that need to be saved including:
- A sample of images and their reconstruction
- The loss curve
- The images used
- The model

#### Image reconstructions
A sample of the images and their reconstructions is needed for the user to judge qualitatively the strength of the reproduction. The reproductions and the originals are fed into the function along with the number of samples. The original and sample are multiplied by 255 and converted to unsigned integers to undo the normalisation performed during preprocessing. each original and reconstructed image is then stacked horizontally and each sample is stacked vertically. The function then outputs the new array which is saved to a file using the cv2 package. You can see the code for this below.

```
def show_predictions(decoded: np.ndarray, gt: np.ndarray, samples: int=10) -> np.ndarray:
    for sample in range(0, samples):
        original = (gt[sample] * 255).astype("uint8")
        reconstructed = (decoded[sample] * 255).astype("uint8")
        output = np.hstack([original, reconstructed])
        if sample == 0:
            outputs = output
        else:
            outputs = np.vstack([outputs, output])
    return outputs
vis = show_predictions(decoded, test_x)
cv2.imwrite(os.path.join("reconstruction", f"{date_time}.png"), vis)
```
The image below shows an example of the image reconstruction.

![Image reconstruction](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/9f479516-ab8b-42df-8a38-4a0e066cbb35)


#### Loss curve
The loss curve helps the user determine which epoch the loss levels out to avoid overfitting. It includes lines representing the training data, which the model is trained on, and the test or validation data, which is used to show how the model performs on data it has not seen. For the model I ended up using, we can see that both sets of data stabilise around 125 epochs.

You can see the code to generate this graph below and the graph itself below.

```
N = np.arange(0, Epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, model.history["loss"], label = "train_loss")
plt.plot(N, model.history["val_loss"], label = "val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(os.path.join("loss",f"{date_time}.jpg"))
```

![21-02-2024-10-45-33](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/79f27f2b-5f9a-4aa4-aff5-1fb415204f59)


#### Images used
The data used to train the model is needed in the review stage. Because of this, I saved the data as a pickle to be extracted by another file.

```
print("saving data")
dataset = os.path.join("output",f"{date_time}.pickle")
f = open(dataset, 'wb')
f.write(pickle.dumps(images))
f.close()
```

#### Model
The state of the model is also needed in the review stage. As such, I used tensorflow's built-in model saver to save the model.

```
print("saving autoencoder")
model_out = os.path.join("model", f"{date_time}.model")
autoencoder.save(model_out, save_format='h5')
```

## Autoencoder
An autoencoder is a neural network that attempts to recreate the image it was initially given. This is achieved by passing the image through an encoder and then passing the output back through a decoder to produce a new image. Due to the informed guess and check nature of building a convolutional neural network model, I wanted to make sure the autoencoder was as modular as possible. As such, the function accepts parameters for the image height, width and depth(number of channels), the number of nodes in each convolutional layer, the window and stride for tiling, and the latent dimension for the dense layer of the encoder.

### Encoder
The encoder starts with an input layer based on the shape of the image. The code then cycles through each node creating a convolutional layer for each based on the tiling parameters with relu activation. After the final node, the network is flattened to be fed into a dense layer based on the latent dimensions input. This final output along with the input is then saved as the encoder model. You can see the code for this below.

```
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from typing import Tuple

class Autoencoder():
    def build(height:int, width:int, depth:int, nodes: Tuple[int], window: int, stride: int, latent_dim: int) -> Model:
        shape = (width, height, depth)
        inputs = layers.Input(shape)
        x = inputs
        for node in nodes:
            x = layers.Conv2D(node, (window,window), activation='relu', padding='same', strides=stride)(x)
        conv_shape = x.shape
        x = layers.Flatten()(x)
        x = layers.Dense(latent_dim + latent_dim)(x)
        latent = x
        encoder = Model(inputs, x, name = 'encoder')
```

### Decoder
The decoder starts with a Dense layer with the number of nodes equal to multiplying the shape of the last convolutional layer of the encoder network. This is to ensure there are enough nodes to rebuild the network. I then added a reshape layer to have the data in the right shape. The nodes are then traversed in reverse with each node adding a transposed convolutional layer with the same activation, window and stride as before. For the output, we use a convolutional layer with sigmoid activation and the number of nodes equal to the image depth. The decoder model is then saved with the encoder's outputs as its inputs and the current output as its output. With both the encoder and decoder now built, we can save the autoencoder with the initial inputs as its input and the result of running the decoder on the encoder's outputs as its outputs. This new autoencoder model is now returned by the function. You can see this second half of the function's code below.

```
        x = layers.Dense(conv_shape[1]*conv_shape[2]*conv_shape[3])(x)
        x = layers.Reshape(target_shape = conv_shape[1:])(x)
        reverse_nodes = nodes[::-1]
        for node in reverse_nodes:
            x = layers.Conv2DTranspose(node, kernel_size=window, strides=stride, activation="relu", padding="same")(x)
        x = layers.Conv2D(depth, kernel_size = (window, window), activation="sigmoid", padding="same")(x)
        outputs = x
        decoder = Model(latent, outputs, name = 'decoder')
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        return autoencoder
```

## Review Results
The file that reviews the results of the autoencoder can be split into four parts:
- Load the data/model
- Calculate errors
- Find the most anomalous images
- Share results

### Load data and model
When we ran the training file, the model, images and labels were saved to separate files. These files are now loaded to assess the results. The code for this is shown below:

```
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import random
from extract import extract_images
import cv2
import os

timestamp = "21-02-2024-10-45-33"
dataset = os.path.join("output",f"{timestamp}.pickle")
model = os.path.join("model",f"{timestamp}.model")
anomaly_file = os.path.join("labels", f"{timestamp}.txt")

autoencoder = load_model(model)
images = pickle.loads(open(dataset, "rb").read())
anomaly_indexes = []
with open(anomaly_file, 'r') as f:
    for anomaly_index in f.readlines():
        anomaly_indexes.append(int(anomaly_index))
```

### Calculate errors
The next step is to calculate the overall and by-pixel errors in each image. The overall mean squared error allows us to identify the worst reconstructed images. The error by pixel allows us to create a heatmap showing the worst-performing areas of each image. After running the images through the model's prediction function, we calculate the overall mean squared error, stored as a float and the mean squared error for each pixel, stored as a numpy array. You can see the code for this section below.

```
decoded = autoencoder.predict(images)
errors = []
pxl_errors = []

for (image, reconstruction) in zip(images, decoded):
    if len(image.shape) == len(reconstruction.shape) + 1:
        image = image.reshape(image.shape[:-1])
    mse = np.mean((image - reconstruction)**2)
    pxl_error = np.mean((image - reconstruction)**2, axis=2)
    errors.append(mse)
    pxl_errors.append(pxl_error)
```

### Finding the most anomalous images
Now that we have a list of the overall error for each image, we can use this to find the images most likely to have defects. We first use numpy to find the 98th quantile (assuming normal distribution) of the errors. This means that anything above this will be in the top 2% of errors. A new list of indexes is created along with the overall and by-pixel errors for any error above this threshold. The code for this section is shown below.

```
threshold = np.quantile(errors, 0.98)
anomalies = np.where(np.array(errors) >= threshold)[0]
anomaly_errors = [error for error in errors if error >= threshold]
anomaly_pxl_errors = [pxl_error for pxl_error, error in zip(pxl_errors,errors) if error>=threshold]
print(f"mse threshold: {threshold}")
```

### Share results
Each image on this new list is checked against its label to see if it was one of the defect images. The results show that with the threshold set to 98%, the model finds 6 of the 7 defect images. For a more qualitative assessment, the code also writes the original, reproduction, and heatmap for each image in the error range to a new file. after some preprocessing, the heatmap is generated using the `applyColormap` function of the cv2 package. The original, reconstruction and heatmap are then stacked horizontally for each image and each image is stacked vertically. This output is then written to a file using cv2. This is done using the code below.

```
anomaly_count = len(anomaly_indexes)
print(f"{len(anomalies)} anomalies found of {len(images)} images and {anomaly_count} anomalies")
sum = 0
for anomaly in anomalies:
    if anomaly in anomaly_indexes:
        sum+=1
print(f"success rate {sum} out of {anomaly_count}")

outputs = None
for error, anomaly, pxl_error in sorted(zip(anomaly_errors, anomalies, anomaly_pxl_errors), reverse=True):
    original = (images[anomaly] * 255).astype("uint8")
    reconstruction = (decoded[anomaly] * 255).astype("uint8")
    heatmap = np.expand_dims(np.round((pxl_error/np.max(pxl_error))*255), axis=-1).astype('uint8')
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    output = np.hstack([original, reconstruction,
                         heatmap_img
                         ])
    if outputs is None:
        outputs = output
    else:
        outputs = np.vstack([outputs, output])
    print(anomaly in anomaly_indexes)
cv2.imwrite(os.path.join("anomalies",f"{timestamp}.png"), outputs)
```

An example of the images alongside the heatmaps is shown below. We can see examples of stringy filament being highlighted as well as the user's hand when they move it into the shot. The wheels of the extruder also appear to raise errors in some images. Taking images less often, whenever the extruder is in a specific position may lead to fewer errors.

![21-02-2024-10-45-33](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/9f9111cb-84f0-4f38-a05b-5947e02638ba)

If you would like to dive more into this project, the code is available at [https://github.com/SamMatt87/3d-printing-anomalies].


## [Return Home](https://sammatt87.github.io/)

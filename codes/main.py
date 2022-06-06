##########                                                          ###########
##########      Read all the necessary packages and libraries       ###########
##########                                                          ###########

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import keras
import itertools
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import StratifiedKFold

##########                                                          ###########
##########                   Necessary Functions                    ###########
##########                                                          ###########

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


##########                                                          ###########
##########                          Main                            ###########
##########                                                          ###########

os.chdir('path/to/scriptsdirectory/')

# includes FPN-based VGG16, ResNet50, DenseNet121, and EfficientNetB0
import baseModels

# includes FPN-based VGG16 with different combination of scales
# top-5: merges all convolutional blocks of the VGG16 model to make a prediction
# top-4: merges top 4 convolutional blocks ...
# top-3: merges top 3 ...
# top-2: merges top 2 ...
# top-1: uses the last convolutional block
import vggCombinations

score_noload = np.zeros([5,2]) 
score_load   = np.zeros([5,2]) 

# parameters
num_classes      = 3
imageSize        = 224
val_dropout      = 0.5
batch_size       = 4
epochs           = 60
weight_init      = 'imagenet'

# create generator
train_datagen = ImageDataGenerator(samplewise_center=True,
                                   samplewise_std_normalization=True,                                   
                                   rotation_range=10,
                                   shear_range=10,
                                   brightness_range=[0.8,1.2],
                                   zoom_range=[0.8,1.2],
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   validation_split= 0.25)

# create generator
test_datagen = ImageDataGenerator(samplewise_center=True,
                                  samplewise_std_normalization=True)

dataframe = pd.read_csv('path/to/datasetdataframe.csv')

# creating labels for the dataset
temp_normal = np.zeros(120)
temp_drusen = np.ones(153)
temp_cnv    = 2*np.ones(154)
temp        = np.concatenate((temp_normal, temp_drusen, temp_cnv))

# stratified k-fold based on patients' images
kfold = StratifiedKFold(n_splits=5, shuffle= True, random_state= 43)

fold = 0

for train_index, test_index in kfold.split(temp, temp):        

    NAME = 'fpn-vgg16-fold%d-{}'.format(int(time.time())) %(fold+1)    

    train_class = []
    for num in train_index:
        train_class.append(dataframe.iloc[np.where(dataframe['Index']== num)[0][0]]['Class'])
    
    train_index, valid_index, train_class, valid_class = train_test_split(train_index, 
                                                                          train_class, 
                                                                          test_size=0.2, 
                                                                          stratify=train_class)
    # get train indices for reading data from dataframe
    train_index_image = []
    for count1 in range(len(train_index)):
        train_index_image.append(np.where(dataframe['Index'] == train_index[count1]))
    train_index_image = np.squeeze(np.hstack(train_index_image).transpose())
    # get validation indices for reading data from dataframe
    valid_index_image = []
    for count1 in range(len(valid_index)):
        valid_index_image.append(np.where(dataframe['Index'] == valid_index[count1]))
    valid_index_image = np.squeeze(np.hstack(valid_index_image).transpose())
    # get test indices for reading data from dataframe
    test_index_image  = []
    for count2 in range(len(test_index)):
        test_index_image.append(np.where(dataframe['Index'] == test_index[count2]))
    test_index_image  = np.squeeze(np.hstack(test_index_image).transpose())

    # get train, validation, and test dataframes    
    trainData = dataframe.iloc[train_index_image]
    validData = dataframe.iloc[valid_index_image]
    testData  = dataframe.iloc[test_index_image]
    print('Initializing fold %s' %str(fold))
    print('Train shape:',trainData.shape)
    print('Valid shape:',validData.shape)
    print('Test shape:',testData.shape)
    
    # training iterator
    train_it=train_datagen.flow_from_dataframe(
    dataframe=trainData,
    x_col="PATH",
    y_col="Class",
    batch_size=batch_size,
    class_mode="categorical",
    target_size=(imageSize, imageSize))
    # validation iterator
    valid_it=train_datagen.flow_from_dataframe(
        dataframe=validData,
        x_col="PATH",
        y_col="Class",
        batch_size=batch_size,
        class_mode="categorical",
        target_size=(imageSize, imageSize))
    # testing iterator
    test_it=test_datagen.flow_from_dataframe(
        dataframe=testData,
        directory=None,
        x_col="PATH",
        y_col="Class",
        batch_size=1,
        shuffle=False,
        class_mode="categorical",
        target_size=(imageSize, imageSize))
    
    # get dataset size in each fold
    nb_train_samples      = train_it.n
    nb_validation_samples = valid_it.n
    nb_test_samples       = test_it.n

    # loading the model
    model = baseModels.VGG16_FPN(num_classes, imageSize, weight_init, val_dropout)
    #model = baseModels.ResNet50_FPN(num_classes, imageSize, weight_init, val_dropout)
    #model = baseModels.DenseNet121_FPN(num_classes, imageSize, weight_init, val_dropout)
    #model = baseModels.EfficientNetB0_FPN(num_classes, imageSize, weight_init, val_dropout)

    # selecting the optimizer
    optimizer1 = keras.optimizers.Adam(lr=1e-4)
    optimizer2 = keras.optimizers.RMSprop(lr=1e-4)
    optimizer3 = keras.optimizers.SGD(lr=1e-4)
    
    # compiling the model
    model.compile(loss='categorical_crossentropy', 
                optimizer=optimizer1, 
                metrics=['accuracy'])
    
    # setting the callbacks
    callbacks = [
      EarlyStopping(monitor='val_loss', patience=10, verbose=1),
      ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1),
      ModelCheckpoint('5.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True),
      TensorBoard(log_dir='logs\\{}'.format(NAME))
      ]

    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_it.classes), train_it.classes)
    class_weights = {i : class_weights[i] for i in range(num_classes)}

    history = model.fit_generator(
                  train_it,   
                  steps_per_epoch=nb_train_samples // (batch_size),
                  epochs=epochs,
                  validation_data=valid_it,
                  validation_steps=nb_validation_samples // (batch_size),
                  class_weight= class_weights,
                  callbacks= callbacks)

    # evaluate model (on the latest model from the last epoch of training)
    score_noload[fold,:] = model.evaluate_generator(test_it,
                                          steps = nb_test_samples // 1,
                                          verbose=1)

    print('\nKeras CNN - accuracy:', score_noload[fold,1], '\n')

    model.load_weights('path/to/model.h5')

    # Evaluate model (on the best trained model based on validation loss)
    score_load[fold,:] = model.evaluate_generator(test_it,
                                        steps = nb_test_samples // 1,
                                        verbose=1)

    print('\nKeras CNN - accuracy:', score_load[fold,1], '\n')
  
    fold += 1
    
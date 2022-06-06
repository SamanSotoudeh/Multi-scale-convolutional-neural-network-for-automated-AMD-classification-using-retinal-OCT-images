##########                                                          ###########
##########      Read all the necessary packages and libraries       ###########
##########                                                          ###########

import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D, UpSampling2D, Add, Concatenate
from tensorflow.keras.applications.vgg16 import VGG16

# initializers
kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

#################                                            ##################
#################        VGG16 Backbone + FPN (TOP 5)        ##################
#################                                            ##################

def VGG16_FPN_top5(num_classes, imageSize, weight_init, val_dropout):
    
    pretrained_model = VGG16(weights = weight_init, 
                             include_top=False, 
                             input_shape=(imageSize, imageSize, 3))

    c5 = pretrained_model.output

    # Block 5
    p5 = Conv2D(256, (1,1), activation='relu', padding='same', name='block5_conv1x1')(c5)
    
    # Block 4
    p4 = Add(name='lateral_block4')([
            UpSampling2D(size=(2,2), name='block5_upsampled')(p5),
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block4_conv1x1')(pretrained_model.layers[-5].output)])
    
    # Block 3
    p3 = Add(name='lateral_block3')([
            UpSampling2D(size=(2,2), name='block4_upsampled')(p4),
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block3_conv1x1')(pretrained_model.layers[-9].output)])

    # Block 2
    p2 = Add(name='lateral_block2')([
            UpSampling2D(size=(2,2), name='block3_upsampled')(p3),
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block2_conv1x1')(pretrained_model.layers[-13].output)])

    # Block 1
    p1 = Add(name='lateral_block1')([
            UpSampling2D(size=(2,2), name='block2_upsampled')(p2),
            Conv2D(256, (1,1), activation='relu', padding='same',
                   name='block1_conv1x1')(pretrained_model.layers[-16].output)])
    
    p5_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p5_dealiasing_conv1')(p5)
    p5_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p5_dealiasing_conv2')(p5_dealiasing)

    p4_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p4_dealiasing_conv1')(p4)
    p4_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p4_dealiasing_conv2')(p4_dealiasing)
    
    p3_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p3_dealiasing_conv1')(p3)
    p3_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p3_dealiasing_conv2')(p3_dealiasing)
    
    p2_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p2_dealiasing_conv1')(p2)
    p2_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p2_dealiasing_conv2')(p2_dealiasing)
    
    p1_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p1_dealiasing_conv1')(p1)
    p1_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p1_dealiasing_conv2')(p1_dealiasing)
    
    p5_pool = GlobalAveragePooling2D(name='gap_p5')(p5_dealiasing)
    p4_pool = GlobalAveragePooling2D(name='gap_p4')(p4_dealiasing)
    p3_pool = GlobalAveragePooling2D(name='gap_p3')(p3_dealiasing)
    p2_pool = GlobalAveragePooling2D(name='gap_p2')(p2_dealiasing)
    p1_pool = GlobalAveragePooling2D(name='gap_p1')(p1_dealiasing)

    gap_all = Concatenate()([p5_pool, p4_pool, p3_pool, p2_pool, p1_pool])
    
    dense1  = Dense(512, activation='relu')(gap_all)
    dense1  = Dropout(val_dropout)(dense1)
    
    pred   = Dense(num_classes, activation= 'softmax')(dense1)
    
    model = Model(inputs= pretrained_model.input, outputs=pred)

    return model


#################                                            ##################
#################        VGG16 Backbone + FPN (TOP 4)        ##################
#################                                            ##################

def VGG16_FPN_top4(num_classes, imageSize, weight_init, val_dropout):
    
    pretrained_model = VGG16(weights = weight_init, 
                             include_top=False, 
                             input_shape=(imageSize, imageSize, 3))

    c5 = pretrained_model.output

    # Block 5
    p5 = Conv2D(256, (1,1), activation='relu', padding='same', name='block5_conv1x1')(c5)
    
    # Block 4
    p4 = Add(name='lateral_block4')([
            UpSampling2D(size=(2,2), name='block5_upsampled')(p5),
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block4_conv1x1')(pretrained_model.layers[-5].output)])
    
    # Block 3
    p3 = Add(name='lateral_block3')([
            UpSampling2D(size=(2,2), name='block4_upsampled')(p4),
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block3_conv1x1')(pretrained_model.layers[-9].output)])

    # Block 2
    p2 = Add(name='lateral_block2')([
            UpSampling2D(size=(2,2), name='block3_upsampled')(p3),
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block2_conv1x1')(pretrained_model.layers[-13].output)])

    
    p5_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p5_dealiasing_conv1')(p5)
    p5_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p5_dealiasing_conv2')(p5_dealiasing)

    p4_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p4_dealiasing_conv1')(p4)
    p4_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p4_dealiasing_conv2')(p4_dealiasing)
    
    p3_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p3_dealiasing_conv1')(p3)
    p3_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p3_dealiasing_conv2')(p3_dealiasing)
    
    p2_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p2_dealiasing_conv1')(p2)
    p2_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p2_dealiasing_conv2')(p2_dealiasing)
    
    p5_pool = GlobalAveragePooling2D(name='gap_p5')(p5_dealiasing)
    p4_pool = GlobalAveragePooling2D(name='gap_p4')(p4_dealiasing)
    p3_pool = GlobalAveragePooling2D(name='gap_p3')(p3_dealiasing)
    p2_pool = GlobalAveragePooling2D(name='gap_p2')(p2_dealiasing)

    gap_all = Concatenate()([p5_pool, p4_pool, p3_pool, p2_pool])
    
    dense1  = Dense(512, activation='relu')(gap_all)
    dense1  = Dropout(val_dropout)(dense1)
    
    pred   = Dense(num_classes, activation= 'softmax')(dense1)
    
    model = Model(inputs= pretrained_model.input, outputs=pred)

    return model


#################                                            ##################
#################        VGG16 Backbone + FPN (TOP 3)        ##################
#################                                            ##################

def VGG16_FPN_top3(num_classes, imageSize, weight_init, val_dropout):
    
    pretrained_model = VGG16(weights = weight_init, 
                             include_top=False, 
                             input_shape=(imageSize, imageSize, 3))

    c5 = pretrained_model.output

    # Block 5
    p5 = Conv2D(256, (1,1), activation='relu', padding='same', name='block5_conv1x1')(c5)
    
    # Block 4
    p4 = Add(name='lateral_block4')([
            UpSampling2D(size=(2,2), name='block5_upsampled')(p5),
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block4_conv1x1')(pretrained_model.layers[-5].output)])
    
    # Block 3
    p3 = Add(name='lateral_block3')([
            UpSampling2D(size=(2,2), name='block4_upsampled')(p4),
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block3_conv1x1')(pretrained_model.layers[-9].output)])
    
    p5_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p5_dealiasing_conv1')(p5)
    p5_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p5_dealiasing_conv2')(p5_dealiasing)

    p4_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p4_dealiasing_conv1')(p4)
    p4_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p4_dealiasing_conv2')(p4_dealiasing)
    
    p3_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p3_dealiasing_conv1')(p3)
    p3_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p3_dealiasing_conv2')(p3_dealiasing)

    p5_pool = GlobalAveragePooling2D(name='gap_p5')(p5_dealiasing)
    p4_pool = GlobalAveragePooling2D(name='gap_p4')(p4_dealiasing)
    p3_pool = GlobalAveragePooling2D(name='gap_p3')(p3_dealiasing)

    gap_all = Concatenate()([p5_pool, p4_pool, p3_pool])
    
    dense1  = Dense(512, activation='relu')(gap_all)
    dense1  = Dropout(val_dropout)(dense1)
    
    pred   = Dense(num_classes, activation= 'softmax')(dense1)
    
    model = Model(inputs= pretrained_model.input, outputs=pred)

    return model

#################                                            ##################
#################        VGG16 Backbone + FPN (TOP 2)        ##################
#################                                            ##################

def VGG16_FPN_top2(num_classes, imageSize, weight_init, val_dropout):
    
    pretrained_model = VGG16(weights = weight_init, 
                             include_top=False, 
                             input_shape=(imageSize, imageSize, 3))

    c5 = pretrained_model.output

    # Block 5
    p5 = Conv2D(256, (1,1), activation='relu', padding='same', name='block5_conv1x1')(c5)
    
    # Block 4
    p4 = Add(name='lateral_block4')([
            UpSampling2D(size=(2,2), name='block5_upsampled')(p5),
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block4_conv1x1')(pretrained_model.layers[-5].output)])
    
    p5_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p5_dealiasing_conv1')(p5)
    p5_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p5_dealiasing_conv2')(p5_dealiasing)

    p4_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p4_dealiasing_conv1')(p4)
    p4_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p4_dealiasing_conv2')(p4_dealiasing)

    p5_pool = GlobalAveragePooling2D(name='gap_p5')(p5_dealiasing)
    p4_pool = GlobalAveragePooling2D(name='gap_p4')(p4_dealiasing)

    gap_all = Concatenate()([p5_pool, p4_pool])
    
    dense1  = Dense(512, activation='relu')(gap_all)
    dense1  = Dropout(val_dropout)(dense1)
    
    pred   = Dense(num_classes, activation= 'softmax')(dense1)
    
    model = Model(inputs= pretrained_model.input, outputs=pred)

    return model

#################                                            ##################
#################        VGG16 Backbone + FPN (TOP 1)        ##################
#################                                            ##################

def VGG16_FPN_top1(num_classes, imageSize, weight_init, val_dropout):
    
    pretrained_model = VGG16(weights = weight_init, 
                             include_top=False, 
                             input_shape=(imageSize, imageSize, 3))

    c5 = pretrained_model.output

    # Block 5
    p5 = Conv2D(256, (1,1), activation='relu', padding='same', name='block5_conv1x1')(c5)
    
    p5_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p5_dealiasing_conv1')(p5)
    p5_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p5_dealiasing_conv2')(p5_dealiasing)

    p5_pool = GlobalAveragePooling2D(name='gap_p5')(p5_dealiasing)
    
    dense1  = Dense(512, activation='relu')(p5_pool)
    dense1  = Dropout(val_dropout)(dense1)
    
    pred   = Dense(num_classes, activation= 'softmax')(dense1)
    
    model = Model(inputs= pretrained_model.input, outputs=pred)

    return model
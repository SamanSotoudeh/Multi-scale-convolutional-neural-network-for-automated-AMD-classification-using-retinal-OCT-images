##########                                                          ###########
##########      Read all the necessary packages and libraries       ###########
##########                                                          ###########

from tensorflow import keras
from keras import Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D, UpSampling2D, Add, Concatenate
import efficientnet.keras as efn
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121

# initializers
kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

#################                                            ##################
#################            VGG16 Backbone + FPN            ##################
#################                                            ##################

def VGG16_FPN(num_classes, imageSize, weight_init, val_dropout):
    
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
#################          ResNet50 Backbone + FPN           ##################
#################                                            ##################

def ResNet50_FPN(num_classes, imageSize, weight_init, val_dropout):
    
    pretrained_model = ResNet50(weights = weight_init, 
                                include_top=False, 
                                input_shape=(imageSize, imageSize, 3))

    c5 = pretrained_model.output

    # Block 5
    p5 = Conv2D(256, (1,1), activation='relu', padding='same', name='block5_conv1x1')(c5)
    
    # Block 4
    p4 = Add(name='lateral_block4')([
            UpSampling2D(size=(2,2), name='block5_upsampled')(p5),
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block4_conv1x1')(pretrained_model.layers[-33].output)])
    
    # Block 3
    p3 = Add(name='lateral_block3')([
            UpSampling2D(size=(2,2), name='block4_upsampled')(p4),
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block3_conv1x1')(pretrained_model.layers[-95].output)])

    # Block 2
    p2 = Add(name='lateral_block2')([
            UpSampling2D(size=(2,2), name='block3_upsampled')(p3),
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block2_conv1x1')(pretrained_model.layers[-137].output)])

    # Block 1
    p1 = Add(name='lateral_block1')([
            UpSampling2D(size=(2,2), name='block2_upsampled')(p2),
            Conv2D(256, (1,1), activation='relu', padding='same',
                   name='block1_conv1x1')(pretrained_model.layers[-171].output)])
    
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
#################         DenseNet121 Backbone + FPN         ##################
#################                                            ##################

def DenseNet121_FPN(num_classes, imageSize, weight_init, val_dropout):
    
    pretrained_model = DenseNet121(weights = weight_init, 
                                   include_top=False, 
                                   input_shape=(imageSize, imageSize, 3))

    c5 = pretrained_model.output

    # Block 5
    p5 = Conv2D(256, (1,1), activation='relu', padding='same', name='block5_conv1x1')(c5)
    
    # Block 4
    p4 = Add(name='lateral_block4')([
            UpSampling2D(size=(2,2), name='block5_upsampled')(p5),
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block4_conv1x1')(pretrained_model.layers[-117].output)])
    
    # Block 3
    p3 = Add(name='lateral_block3')([
            UpSampling2D(size=(2,2), name='block4_upsampled')(p4),
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block3_conv1x1')(pretrained_model.layers[-289].output)])

    # Block 2
    p2 = Add(name='lateral_block2')([
            UpSampling2D(size=(2,2), name='block3_upsampled')(p3),
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block2_conv1x1')(pretrained_model.layers[-377].output)])

    # Block 1
    p1 = Add(name='lateral_block1')([
            UpSampling2D(size=(2,2), name='block2_upsampled')(p2),
            Conv2D(256, (1,1), activation='relu', padding='same',
                   name='block1_conv1x1')(pretrained_model.layers[-423].output)])
    
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
#################        EfficientNetB0 Backbone + FPN       ##################
#################                                            ##################

def EfficientNetB0_FPN(num_classes, imageSize, weight_init, val_dropout):
    
    pretrained_model = efn.EfficientNetB0(weights = weight_init, 
                                          include_top=False, 
                                          input_shape=(imageSize, imageSize, 3))

    c7 = pretrained_model.get_layer('block7a_project_bn').output

    # Block 7
    p7 = Conv2D(256, (1,1), activation='relu', padding='same', name='block7_conv1x1')(c7)
    
    # Block 6
    p6 = Add(name='lateral_block6')([
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block7_conv1x1_add')(p7),
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block6_conv1x1')(pretrained_model.layers[-17].output)])
    
    # Block 5
    p5 = Add(name='lateral_block5')([
            UpSampling2D(size=(2,2), name='block6_upsampled')(p6),
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block5_conv1x1')(pretrained_model.layers[-75].output)])

    # Block 4
    p4 = Add(name='lateral_block4')([
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block5_conv1x1_add')(p5),
            Conv2D(256, (1,1), activation='relu', padding='same', 
                   name='block4_conv1x1')(pretrained_model.layers[-118].output)])

    # Block 3
    p3 = Add(name='lateral_block3')([
            UpSampling2D(size=(2,2), name='block4_upsampled')(p4),
            Conv2D(256, (1,1), activation='relu', padding='same',
                   name='block3_conv1x1')(pretrained_model.layers[-161].output)])

    # Block 2
    p2 = Add(name='lateral_block2')([
            UpSampling2D(size=(2,2), name='block3_upsampled')(p3),
            Conv2D(256, (1,1), activation='relu', padding='same',
                   name='block2_conv1x1')(pretrained_model.layers[-189].output)])

    # Block 1
    p1 = Add(name='lateral_block1')([
            UpSampling2D(size=(2,2), name='block2_upsampled')(p2),
            Conv2D(256, (1,1), activation='relu', padding='same',
                   name='block1_conv1x1')(pretrained_model.layers[-217].output)])

    p7_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p7_dealiasing_conv1')(p7)
    p7_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p7_dealiasing_conv2')(p7_dealiasing)
    
    p6_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p6_dealiasing_conv1')(p6)
    p6_dealiasing = Conv2D(256, (3,3), activation='relu', padding='same', name='p6_dealiasing_conv2')(p6_dealiasing)
    
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
    
    p7_pool = GlobalAveragePooling2D(name='gap_p7')(p7_dealiasing)
    p6_pool = GlobalAveragePooling2D(name='gap_p6')(p6_dealiasing)    
    p5_pool = GlobalAveragePooling2D(name='gap_p5')(p5_dealiasing)
    p4_pool = GlobalAveragePooling2D(name='gap_p4')(p4_dealiasing)
    p3_pool = GlobalAveragePooling2D(name='gap_p3')(p3_dealiasing)
    p2_pool = GlobalAveragePooling2D(name='gap_p2')(p2_dealiasing)
    p1_pool = GlobalAveragePooling2D(name='gap_p1')(p1_dealiasing)

    gap_all = Concatenate()([p7_pool, p6_pool, p5_pool, p4_pool, p3_pool, p2_pool, p1_pool])
    
    dense1  = Dense(256, activation='relu')(gap_all)
    dense1  = Dropout(val_dropout)(dense1)
    dense2  = Dense(128, activation='relu')(dense1)
    dense2  = Dropout(val_dropout)(dense2)
    
    pred   = Dense(num_classes, activation= 'softmax')(dense2)
    
    model = Model(inputs= pretrained_model.input, outputs=pred)

    return model

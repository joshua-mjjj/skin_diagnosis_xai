from keras.layers import Input, Lambda, Dense, Flatten, Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras_tuner.tuners import RandomSearch
import keras_tuner
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE = [224, 224]

# Dataset path
train_path = 'Data/train'
test_path = 'Data/test'

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

vgg.input

for layer in vgg.layers:
  layer.trainable = False

from keras import optimizers
from keras.regularizers import l1, l2

BACTH_SIZE = 20 

def build_model(hp): 
    x = Flatten()(vgg.output)

    dropout_rate  = hp.Float('dropout_rate',  min_value=0.01, max_value=0.2, step=0.01)  
    dropout_rate1 = hp.Float('dropout_rate1', min_value=0.01, max_value=0.2, step=0.01)
    dropout_rate2 = hp.Float('dropout_rate2', min_value=0.01, max_value=0.2, step=0.01)
    dropout_rate3 = hp.Float('dropout_rate3', min_value=0.01, max_value=0.2, step=0.01)

    l2_strength = hp.Choice('l2_strength', values=[0.001, 0.01, 0.1, 0.15, 0.2])
    l1_strength = hp.Choice('l1_strength', values=[0.001, 0.01, 0.1, 0.15, 0.2])
    
    x = Dense(
        units=hp.Int('units', min_value=128, max_value=1024, step=64),  
        activation='relu',
        kernel_initializer='he_uniform',
        kernel_regularizer=l1(l1_strength) 
    )(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(
        units=hp.Int('units_1', min_value=128, max_value=832, step=64),  
        activation='relu',
        kernel_initializer='he_uniform',
        kernel_regularizer=l1(l1_strength) 
    )(x)
    x = Dropout(dropout_rate1)(x)

    x = Dense(
        units=hp.Int('units_2', min_value=128, max_value=832, step=64),  
        activation='relu',
        kernel_initializer='he_uniform',
        kernel_regularizer=l2(l2_strength) 
    )(x)
    x = Dropout(dropout_rate2)(x)

    x = Dense(
        units=hp.Int('units_3', min_value=128, max_value=832, step=64),  
        activation='relu',
        kernel_initializer='he_uniform',
        kernel_regularizer=l2(l2_strength) 
    )(x)
    x = Dropout(dropout_rate3)(x)

    # num_layers = hp.Int('num_layers', min_value=2, max_value=7, step=1)
    # for i in range(num_layers):
    #     x = Dense(
    #         units=hp.Int('units_' + str(i), min_value=128, max_value=1024, step=64),
    #         activation='relu',
    #         kernel_initializer='he_uniform',
    #         kernel_regularizer=l2(l2_strength)
    #     )(x)
    #     x = Dropout(rate=dropout_rate)(x)

    # 5 classes
    prediction = Dense(5, activation='softmax')(x) 
    model = Model(inputs=vgg.input, outputs=prediction)

    model.compile(
        optimizer=optimizers.Adam(
            hp.Choice('learning_rate', [1e-1, 1e-2, 1e-3, 1e-4]),
        ),
        metrics=['accuracy'],
        loss='categorical_crossentropy'
    )

    return model

tuner = RandomSearch(
    build_model,
    objective=keras_tuner.Objective("accuracy", direction="max"),
    max_trials=10,
    # overwrite=True,
    executions_per_trial=3,
    directory="Results",
    project_name="Skin"
)

# Space summary
tuner.search_space_summary()

train_datagen = ImageDataGenerator(
	    rescale=1./255,
	    shear_range=0.05,
	    zoom_range=0.05,
	    horizontal_flip=True,
	    preprocessing_function=preprocess_input,
  )

test_datagen = ImageDataGenerator(
  	rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    preprocessing_function=preprocess_input,
  )

train_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = BACTH_SIZE,
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = BACTH_SIZE,
                                            class_mode = 'categorical')

train_data, train_labels = next(train_set)
validation_data, validation_labels = next(test_set)

tuner.search(train_data, train_labels,
             epochs=100,
             validation_data=(validation_data, validation_labels))  

tuner.results_summary()
models = tuner.get_best_models()


# Results
# Best accuracy So Far: 1.0
# Total elapsed time: 02h 28m 33s
# INFO:tensorflow:Oracle triggered exit
# Results summary
# Results in Results/Skin
# Showing 10 best trials
# Objective(name="accuracy", direction="max")

# Trial 02 summary
# Hyperparameters:
# dropout_rate: 0.09999999999999999
# dropout_rate1: 0.09999999999999999
# dropout_rate2: 0.14
# dropout_rate3: 0.05
# l2_strength: 0.2
# l1_strength: 0.001
# units: 1024
# units_1: 640
# units_2: 832
# units_3: 512
# learning_rate: 0.0001
# Score: 1.0

# Trial 03 summary
# Hyperparameters:
# dropout_rate: 0.08
# dropout_rate1: 0.04
# dropout_rate2: 0.09999999999999999
# dropout_rate3: 0.02
# l2_strength: 0.15
# l1_strength: 0.001
# units: 1024
# units_1: 256
# units_2: 512
# units_3: 448
# learning_rate: 0.001
# Score: 1.0

# Trial 09 summary
# Hyperparameters:
# dropout_rate: 0.15000000000000002
# dropout_rate1: 0.03
# dropout_rate2: 0.01
# dropout_rate3: 0.04
# l2_strength: 0.15
# l1_strength: 0.15
# units: 512
# units_1: 320
# units_2: 448
# units_3: 384
# learning_rate: 0.001
# Score: 0.9166666467984518

# Trial 00 summary
# Hyperparameters:
# dropout_rate: 0.16
# dropout_rate1: 0.08
# dropout_rate2: 0.18000000000000002
# dropout_rate3: 0.05
# l2_strength: 0.01
# l1_strength: 0.2
# units: 576
# units_1: 512
# units_2: 192
# units_3: 768
# learning_rate: 0.001
# Score: 0.8166666626930237

# Trial 07 summary
# Hyperparameters:
# dropout_rate: 0.04
# dropout_rate1: 0.19
# dropout_rate2: 0.16
# dropout_rate3: 0.09999999999999999
# l2_strength: 0.1
# l1_strength: 0.1
# units: 896
# units_1: 192
# units_2: 512
# units_3: 448
# learning_rate: 0.0001
# Score: 0.7833333412806193

# Trial 05 summary
# Hyperparameters:
# dropout_rate: 0.14
# dropout_rate1: 0.13
# dropout_rate2: 0.04
# dropout_rate3: 0.12
# l2_strength: 0.1
# l1_strength: 0.1
# units: 192
# units_1: 128
# units_2: 256
# units_3: 640
# learning_rate: 0.0001
# Score: 0.7333333492279053

# Trial 06 summary
# Hyperparameters:
# dropout_rate: 0.17
# dropout_rate1: 0.13
# dropout_rate2: 0.12
# dropout_rate3: 0.19
# l2_strength: 0.001
# l1_strength: 0.2
# units: 640
# units_1: 832
# units_2: 576
# units_3: 576
# learning_rate: 0.01
# Score: 0.5166666706403097

# Trial 01 summary
# Hyperparameters:
# dropout_rate: 0.12
# dropout_rate1: 0.060000000000000005
# dropout_rate2: 0.08
# dropout_rate3: 0.18000000000000002
# l2_strength: 0.2
# l1_strength: 0.1
# units: 512
# units_1: 256
# units_2: 384
# units_3: 512
# learning_rate: 0.1
# Score: 0.48333332935969037

# Trial 08 summary
# Hyperparameters:
# dropout_rate: 0.06999999999999999
# dropout_rate1: 0.06999999999999999
# dropout_rate2: 0.19
# dropout_rate3: 0.11
# l2_strength: 0.001
# l1_strength: 0.1
# units: 128
# units_1: 256
# units_2: 512
# units_3: 384
# learning_rate: 0.1
# Score: 0.44999998807907104

# Trial 04 summary
# Hyperparameters:
# dropout_rate: 0.04
# dropout_rate1: 0.01
# dropout_rate2: 0.08
# dropout_rate3: 0.16
# l2_strength: 0.2
# l1_strength: 0.1
# units: 1024
# units_1: 128
# units_2: 640
# units_3: 640
# learning_rate: 0.1
# Score: 0.4333333373069763
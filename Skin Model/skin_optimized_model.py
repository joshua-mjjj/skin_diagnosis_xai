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

    dropout_rate  = hp.Float('dropout_rate',  min_value=0.02, max_value=0.1, step=0.02)  
    dropout_rate1 = hp.Float('dropout_rate1', min_value=0.02, max_value=0.15, step=0.02)
    dropout_rate2 = hp.Float('dropout_rate2', min_value=0.02, max_value=0.1, step=0.02)
    dropout_rate3 = hp.Float('dropout_rate3', min_value=0.02, max_value=0.15, step=0.02)

    l2_strength = hp.Choice('l2_strength', values=[0.001, 0.01, 0.1]) 
    l1_strength = hp.Choice('l1_strength', values=[0.001, 0.01, 0.1])
    
    x = Dense(
        units=hp.Int('units', min_value=128, max_value=1024, step=64),  
        activation='relu',
        kernel_initializer='he_uniform',
        kernel_regularizer=l1(l1_strength) 
    )(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(
        units=hp.Int('units_1', min_value=64, max_value=1024, step=64),  
        activation='relu',
        kernel_initializer='he_uniform',
        kernel_regularizer=l1(l1_strength) 
    )(x)
    x = Dropout(dropout_rate1)(x)

    x = Dense(
        units=hp.Int('units_2', min_value=64, max_value=1024, step=64),  
        activation='relu',
        kernel_initializer='he_uniform',
        kernel_regularizer=l2(l2_strength) 
    )(x)
    x = Dropout(dropout_rate2)(x)

    x = Dense(
        units=hp.Int('units_3', min_value=64, max_value=1024, step=64),  
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
    objective=keras_tuner.Objective("val_accuracy", direction="max"),
    max_trials=10,
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
             epochs=80,
             validation_data=(validation_data, validation_labels))  

tuner.results_summary()
models = tuner.get_best_models()

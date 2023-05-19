from keras.layers import Input, Lambda, Dense, Flatten, Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch
import keras_tuner
import keras
import warnings

from keras import optimizers
from keras.regularizers import l1, l2


warnings.filterwarnings("ignore", category=FutureWarning)

IMAGE_SIZE = [224, 224]

train_path = 'Data/train'
test_path = 'Data/test'

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

vgg.input

for layer in vgg.layers:
  layer.trainable = False

# Bare model
EPOCHS = 100

x = Flatten()(vgg.output)

x = Dense(1024, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l1(0.2) )(x)
x = Dropout(0.09)(x)

x = Dense(640, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l1(0.2))(x)
x = Dropout(0.09)(x)

x = Dense(832, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.14)(x)

x = Dense(512, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.05)(x)

prediction = Dense(5, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()

model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=0.0001),
      metrics=['accuracy'],
      loss='binary_crossentropy'
  )


train_datagen = ImageDataGenerator(
      rescale=1./255,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    preprocessing_function=preprocess_input,
    validation_split=0.2
  )

test_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    preprocessing_function=preprocess_input,
    validation_split=0.2
  )

train_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')

validation_set = train_datagen.flow_from_directory(test_path,
                                                   target_size=(224, 224),
                                                   batch_size=10,
                                                   class_mode='categorical') 

(train_data, train_labels) = next(train_set)
(validation_data, validation_labels) = next(validation_set)

from datetime import datetime
from keras.callbacks import ModelCheckpoint


checkpoint = ModelCheckpoint(filepath='skin.h5', monitor='accuracy', mode='max',
                             verbose=2, save_best_only=True)

callbacks = [checkpoint]
start = datetime.now()
model_history = model.fit(
                        train_set,
                        validation_data=validation_set,
                        epochs=EPOCHS,
                        callbacks=callbacks,
                        verbose=1
                       )

duration = datetime.now() - start
print("Training completed in time: ", duration)

# Plotting loss track 
plt.plot(model_history.history['loss'], label='Training loss')
plt.plot(model_history.history['val_loss'], label='Validation loss')
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig("model_loss" + ".png")

print(model_history.history.keys())
# summarize history for accuracy
plt.figure() 
plt.plot(model_history.history['accuracy'], label='Training Accuracy')
plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.savefig("model_accuracy" + ".png")

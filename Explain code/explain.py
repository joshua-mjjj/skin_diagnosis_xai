from keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
from tf_keras_vis.utils.scores import CategoricalScore
from matplotlib import cm




# load and create model
model_path = os.path.join('Model Artifacts', 'Model', 'skin_model_3.h5')
model = tf.keras.models.load_model(model_path)
print("Loaded Skin model from disk...")


# Image titles
image_titles = ['Acne', 'Eczema', 'Melanoma', 'Rosacea', 'Keratoses']

# load img
img_path_1 = os.path.join('Explain', 'Acne', 'Acne.png')
img_path_2 = os.path.join('Explain', 'Eczema', 'Eczema.png')
img_path_3 = os.path.join('Explain', 'Melanoma', 'Melanoma.jpg')
img_path_4 = os.path.join('Explain', 'Rosacea', 'Rosacea.png')
img_path_5 = os.path.join('Explain', 'Seborrheic Keratoses', 'Keratoses.jpg')

img_1 = load_img(img_path_1, target_size=(224,224))
img_2 = load_img(img_path_2, target_size=(224,224))
img_3 = load_img(img_path_3, target_size=(224,224))
img_4 = load_img(img_path_4, target_size=(224,224))
img_5 = load_img(img_path_5, target_size=(224,224))

images = np.asarray(
    [np.array(img_1), 
     np.array(img_2), 
     np.array(img_3), 
     np.array(img_4), 
     np.array(img_5) 
])

# Preparing input data for VGG16
X = preprocess_input(images)

# Rendering
f, ax = plt.subplots(nrows=1, ncols=5, figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].axis('off')
plt.tight_layout()
plt.show()





predict_img_1 = model.predict(preprocess_input(images[0][np.newaxis, ...]))
print(predict_img_1)
print(np.argmax(predict_img_1))

# index of the highest predicted probability
# if np.argmax(predict_img_1) == 0:
#     print("Image 1")
#     print("Benign")
# else:
#     print("Image 1")
#     print("Malignant")

class_idx = np.argsort(predict_img_1.flatten())[::-1]
print(class_idx)

predict_img_2 = model.predict(preprocess_input(images[1][np.newaxis, ...]))
print(predict_img_2)
print(np.argmax(predict_img_2))

# if np.argmax(predict_img_2) == 0:
#     print("Image 2")
#     print("Benign")
# else:
#     print("Image 2")
#     print("Malignant")

class_idx2 = np.argsort(predict_img_2.flatten())[::-1]
print(class_idx2)

predict_img_3 = model.predict(preprocess_input(images[2][np.newaxis, ...]))
print(predict_img_3)
print(np.argmax(predict_img_3))


# if np.argmax(predict_img_3) == 0:
#     print("Image 3")
#     print("Benign")
# else:
#     print("Image 3")
#     print("Malignant")

class_idx3 = np.argsort(predict_img_3.flatten())[::-1]
print(class_idx3)


predict_img_4 = model.predict(preprocess_input(images[3][np.newaxis, ...]))
print(predict_img_4)
print(np.argmax(predict_img_4))


# if np.argmax(predict_img_3) == 0:
#     print("Image 3")
#     print("Benign")
# else:
#     print("Image 3")
#     print("Malignant")

class_idx4 = np.argsort(predict_img_4.flatten())[::-1]
print(class_idx4)


predict_img_5 = model.predict(preprocess_input(images[4][np.newaxis, ...]))
print(predict_img_5)
print(np.argmax(predict_img_5))


# if np.argmax(predict_img_3) == 0:
#     print("Image 3")
#     print("Benign")
# else:
#     print("Image 3")
#     print("Malignant")

class_idx5 = np.argsort(predict_img_5.flatten())[::-1]
print(class_idx5)








# Explaining
print("Explaining Predictions........................")
from tf_keras_vis.saliency import Saliency


# Defining categorical scores
score = CategoricalScore([0, 4])

def model_modifier_function(model):
    model.layers[-1].activation = tf.keras.activations.linear


# Saliency Maps
# Create Saliency object.
saliency = Saliency(model,
                    model_modifier=model_modifier_function,
                    clone=True)

# Generate saliency map
saliency_map = saliency(score, X)

# Render
print("Saliency Maps Visualization...")
f, ax = plt.subplots(nrows=1, ncols=5, figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(saliency_map[i], cmap='jet')
    ax[i].axis('off')
plt.tight_layout()
plt.show()


print("SmoothGrad Visualization...")
# SmoothGrad
# Generate saliency map with smoothing that reduce noise by adding noise
# reduce the impact of noise in the input data on the prediction of the model.
saliency_map = saliency(score,
                        X,
                        smooth_samples=20, # The number of calculating gradients iterations.
                        smooth_noise=0.20) # noise spread level.

# Render
f, ax = plt.subplots(nrows=1, ncols=5, figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(saliency_map[i], cmap='jet')
    ax[i].axis('off')
plt.tight_layout()
# plt.savefig('images/smoothgrad.png')
plt.show()


print("Gradcam Visualization...")
# GradCam
from tf_keras_vis.gradcam import Gradcam

# Defining categorical scores
score = CategoricalScore([0, 1])

# Create Gradcam object
gradcam = Gradcam(model,
                  model_modifier=model_modifier_function,
                  clone=True)

# Generate heatmap with GradCAM
cam_ = gradcam(score, X, penultimate_layer=-1)

# Render
f, ax = plt.subplots(nrows=1, ncols=5, figsize=(12, 4))
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam_[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
    ax[i].axis('off')
plt.tight_layout()
plt.show()



from tf_keras_vis.scorecam import Scorecam
print("Scorecam Visualization (Extends gradCAM)...")
# Create ScoreCAM object
scorecam = Scorecam(model)

predictions = model.predict(X)

# Generate heatmap with ScoreCAM
score_ = CategoricalScore([np.argmax(pred) for pred in predictions])
score_cam = scorecam(score_, X, penultimate_layer=-1, max_N=10)

# Render
f, ax = plt.subplots(nrows=1, ncols=5, figsize=(12, 4))
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(score_cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[i].axis('off')
plt.tight_layout()
plt.show()
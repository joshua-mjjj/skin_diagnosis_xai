from keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
from matplotlib import cm


# # Image titles
# image_titles = ['Acne', 'Acne', 'Acne', 'Acne']
# image_titles_eczema = ['Eczema', 'Eczema', 'Eczema', 'Eczema']
# image_titles_Melanoma = ['Melanoma', 'Melanoma', 'Melanoma', 'Melanoma']
# image_titles_Rosacea = ['Rosacea', 'Rosacea', 'Rosacea', 'Rosacea']
# image_titles_Keratoses = ['Keratoses', 'Keratoses', 'Keratoses', 'Keratoses']


# # load img
# img_path_Acne_1 = load_img(os.path.join('Explain', 'Acne', 'Acne303.png') , target_size=(224,224))
# img_path_Acne_2 = load_img(os.path.join('Explain', 'Acne', 'Acne306.png') , target_size=(224,224))
# img_path_Acne_3 = load_img(os.path.join('Explain', 'Acne', 'Acne343.png') , target_size=(224,224))
# img_path_Acne_4 = load_img(os.path.join('Explain', 'Acne', 'Acne344.png') , target_size=(224,224))

# img_path_Eczema_1 = load_img(os.path.join('Explain', 'Eczema', 'Screenshot from 2023-04-27 11-21-58.png'), target_size=(224,224))
# img_path_Eczema_2 = load_img(os.path.join('Explain', 'Eczema', 'Screenshot from 2023-04-27 11-22-03.png'), target_size=(224,224))
# img_path_Eczema_3 = load_img(os.path.join('Explain', 'Eczema', 'Screenshot from 2023-04-27 11-25-52.png'), target_size=(224,224))
# img_path_Eczema_4 = load_img(os.path.join('Explain', 'Eczema', 'Screenshot from 2023-04-27 11-26-01.png'), target_size=(224,224))

# img_path_Melanoma_1 = load_img(os.path.join('Explain', 'Melanoma', '36.jpg'), target_size=(224,224))
# img_path_Melanoma_2 = load_img(os.path.join('Explain', 'Melanoma', '41.jpg'), target_size=(224,224))
# img_path_Melanoma_3 = load_img(os.path.join('Explain', 'Melanoma', '42.jpg'), target_size=(224,224))
# img_path_Melanoma_4 = load_img(os.path.join('Explain', 'Melanoma', '43.jpg'), target_size=(224,224))

# img_path_Rosacea_1 = load_img(os.path.join('Explain', 'Rosacea', 'Rosacea-112.png'), target_size=(224,224))
# img_path_Rosacea_2 = load_img(os.path.join('Explain', 'Rosacea', 'Rosacea-113.png'), target_size=(224,224))
# img_path_Rosacea_3 = load_img(os.path.join('Explain', 'Rosacea', 'Rosacea-126.png'), target_size=(224,224))
# img_path_Rosacea_4 = load_img(os.path.join('Explain', 'Rosacea', 'Rosacea-127.png'), target_size=(224,224))

# img_path_Keratoses_1 = load_img(os.path.join('Explain', 'Seborrheic Keratoses', '17.jpg'), target_size=(224,224))
# img_path_Keratoses_2 = load_img(os.path.join('Explain', 'Seborrheic Keratoses', '22.jpg'), target_size=(224,224))
# img_path_Keratoses_3 = load_img(os.path.join('Explain', 'Seborrheic Keratoses', '24.jpg'), target_size=(224,224))
# img_path_Keratoses_4 = load_img(os.path.join('Explain', 'Seborrheic Keratoses', '25.jpg'), target_size=(224,224))


# acne_images = np.asarray([np.array(img_path_Acne_1), np.array(img_path_Acne_2), 
#     np.array(img_path_Acne_3), np.array(img_path_Acne_4)])

# eczema_images = np.asarray([np.array(img_path_Eczema_1), np.array(img_path_Eczema_2), 
#     np.array(img_path_Eczema_3), np.array(img_path_Eczema_4)])

# melanoma_images = np.asarray([np.array(img_path_Melanoma_1), np.array(img_path_Melanoma_2), 
#     np.array(img_path_Melanoma_3), np.array(img_path_Melanoma_4)])

# rosacea_images = np.asarray([np.array(img_path_Rosacea_1), np.array(img_path_Rosacea_2), 
#     np.array(img_path_Rosacea_3), np.array(img_path_Rosacea_4)])

# keratoses_images = np.asarray([np.array(img_path_Keratoses_1), np.array(img_path_Keratoses_2), 
#     np.array(img_path_Keratoses_3), np.array(img_path_Keratoses_4)])


# # Preparing input data for VGG16
# X = preprocess_input(acne_images)

# # Rendering
# f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
# for i, title in enumerate(image_titles):
#     ax[i].set_title(title, fontsize=16)
#     ax[i].imshow(acne_images[i])
#     ax[i].axis('off')
# plt.tight_layout()
# plt.show()


# X_1 = preprocess_input(eczema_images)

# # Rendering
# f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
# for i, title in enumerate(image_titles_eczema):
#     ax[i].set_title(title, fontsize=16)
#     ax[i].imshow(eczema_images[i])
#     ax[i].axis('off')
# plt.tight_layout()
# plt.show()

# X_2 = preprocess_input(melanoma_images)

# # Rendering
# f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
# for i, title in enumerate(image_titles_Melanoma):
#     ax[i].set_title(title, fontsize=16)
#     ax[i].imshow(melanoma_images[i])
#     ax[i].axis('off')
# plt.tight_layout()
# plt.show()

# X_3 = preprocess_input(rosacea_images)

# # Rendering
# f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
# for i, title in enumerate(image_titles_Rosacea):
#     ax[i].set_title(title, fontsize=16)
#     ax[i].imshow(rosacea_images[i])
#     ax[i].axis('off')
# plt.tight_layout()
# plt.show()

# X_4 = preprocess_input(keratoses_images)

# # Rendering
# f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
# for i, title in enumerate(image_titles_Keratoses):
#     ax[i].set_title(title, fontsize=16)
#     ax[i].imshow(keratoses_images[i])
#     ax[i].axis('off')
# plt.tight_layout()
# plt.show()



import pandas as pd 
import seaborn as sns

root_dir = 'Data/test'
root_dir_train = 'Data/train'

acne_train_path = os.path.join(root_dir, 'Acne')
eczema_train_path = os.path.join(root_dir, 'Eczema')
melanoma_train_path = os.path.join(root_dir, 'Melanoma')
rosacea_train_path = os.path.join(root_dir, 'Rosacea')
keratoses_train_path = os.path.join(root_dir, 'Seborrheic Keratoses')

acne_train_files = ([files_ for _, _, files_ in os.walk(acne_train_path)])[0]
eczema_train_files = ([files_ for _, _, files_ in os.walk(eczema_train_path)])[0]
melanoma_train_files = ([files_ for _, _, files_ in os.walk(melanoma_train_path)])[0]
rosacea_train_files = ([files_ for _, _, files_ in os.walk(rosacea_train_path)])[0]
keratoses_train_files = ([files_ for _, _, files_ in os.walk(keratoses_train_path)])[0]

final_df = pd.DataFrame()

acne_df = pd.DataFrame()
aczema_df = pd.DataFrame()
melanoma_df = pd.DataFrame()
rosacea_df = pd.DataFrame()
keraroses_df = pd.DataFrame()

acne_df['Image'] =  [acne_train_path+'/'+img for img in acne_train_files]
aczema_df['Image'] =  [eczema_train_path+'/'+img for img in eczema_train_files]
melanoma_df['Image'] =  [melanoma_train_path+'/'+img for img in melanoma_train_files]
rosacea_df['Image'] =  [rosacea_train_path+'/'+img for img in rosacea_train_files]
keraroses_df['Image'] =  [keratoses_train_path+'/'+img for img in keratoses_train_files]

acne_df['Label'] = "acne"
aczema_df['Label'] = "eczema"
melanoma_df['Label'] = "melanoma"
rosacea_df['Label'] = "rosacea"
keraroses_df['Label'] = "keratoses"



final_df = final_df.append([acne_df, aczema_df, melanoma_df, rosacea_df, keraroses_df])

ax = sns.countplot(x=final_df['Label'],
                   order=final_df['Label'].value_counts(ascending=False).index);

abs_values = final_df['Label'].value_counts(ascending=False).values

ax.bar_label(container=ax.containers[0], labels=abs_values);
















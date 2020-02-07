# EfficientNets in Keras
Keras implementation of EfficientNet model for age and gender estimation from the paper
[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

Contains code to build the EfficientNets B0-B7 from the paper, and includes weights for configurations.
![](https://user-images.githubusercontent.com/48142689/73998131-3a93b400-4993-11ea-87f1-e2d0459149a5.png)

In this paper, I used EfficientNets B4 model for estimation.

# Age and Gender estimation
Facial details analysis play an important role in understanding human behavior. Detail such as age and gender 
are the key indicator in analyzing characteristic of a person.

In this paper, we propose an architecture of EfficientNets which can jointly learn representations for three tasks: 
face detection, gender and age classification.

# Dependencies
- Python 3.7+
- EfficientNets library (!!! important: pip install -U efficientnet==0.0.4)
- OpenCV2
- Keras
- Numpy

# Datasets
This is a Keras implementation of a EfficientNet model for estimating age and gender from a face image. In this paper, I use 
[the IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) for training and [the UTKFace dataset](https://susanqq.github.io/UTKFace/)
for testing, which consist of age and gender information

# Create database for training from the IDMB-WIKI Datasets
Dataset consist of most popular 100,000 actors as listed on the IMDb website and (automatically) crawled from their profiles
date of birth, name, gender and all images related to that person. With the additional information from Wikipedia.

IMDB-WIKI age data have over 101 age classes, we want to create a model that is practical for real world application

There are some images in the data are mis labeled (some with negative age), we have manually check and reorganize the data. 
The datasets is processed into 5 classes of age and 2 classes of gender: ***age_branch***(1-13, 14-23, 24-39, 40-55, 56-85), 
***gender_branch***(0: Male, 1: Female) based on facial feature of each age segment

In this model, we used 32000 images for training.

- **Firstly**, dowload the datasets from [the IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

- **Secondly**, filter out noise data and serialize images and labels for training into .mat file. Please click ***data*** folder to
run create_db.py 

# Create database for testing from the UTKFace Datasets
UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of
over 20,000 face images with annotations of age, gender, and ethnicity. Don't use images which have age from 86-100. So the datasets reduce
into 19000 face images

- Dowload the datasets from [the UTKFace dataset](https://susanqq.github.io/UTKFace/). UTKFace.tar.gz can be downloaded 
from Aligned&Cropped Faces in Datasets section
- Run file data/create_db.py

# Preprocess
- To increase accuracy, we used MTCNN to crop the faces and remove non-face image.
- To conveniently moving dataset to Google Drive (for training in Google Collab). We encode all dataset to ‘.npy’ format 
(Numpy array) instead of uploading every images. All the datasets are decoded into images before training the model.

# Train the network
- Please see [age_gender_estimation.ipynb](https://github.com/buiquangmanhhp1999/age_gender_estimation/blob/master/age_gender_estimation.ipynb)

# Result
|  Branch  | Train |  Val  | Test |
|-----|-------|-------|------|
| `AGE`    | 66.71%  | 61.95% | 62% |
| `GENDER` | 96.74%  | 94.13% | 91.76% |

# Use the trained network
`python3 inference_img.py`

- Estimated result: 
![selenagomez](https://user-images.githubusercontent.com/48142689/74002820-ff997c80-49a2-11ea-815b-2c64ffc91193.jpg)

# References
1. Mingxing Tan and Quoc V. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019. Arxiv link: https://arxiv.org/abs/1905.11946.'
2. [https://github.com/qubvel/efficientnet](https://github.com/qubvel/efficientnet)


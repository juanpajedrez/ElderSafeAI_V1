# IoT System for Elderly Fall Detection and Mitigation
This project is part of the AAI-530 course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

**-- Project Status: Completed**

### Installation

To run this project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/wushuchris/ElderSafeAI```
2. Run with Jypiter: https://jupyter.org/install 

## Project Intro/Objective
The main objective of this project is to build a predictive model that is able to obtain video camera footage from different angles (8 cameras), at different scenarios (23 chutes) were fall/no fall events were happening.

### Contributors
- Laurentius Von Liechti
- Christopher Mendoza
- Juan Pablo Triana Martinez

### Methods Used
- Exploratory Data Analysis (EDA) (3D Pose estimation pipeline)
- Exploratory Data Analysis (Optical Flow Gradients)
- Data Cleaning and Preparation
- CNN VGG16 finetuned for Fall Detection.
- CNN Optimized MyLeNet for Fall Detection.
- CNN-LSTM for Fall Prediction
- Model Evaluation
- Data Visualization

### Technologies
- Python
- Jupyter Notebook

## Project Description

#### Dataset
- **Source**: https://www.kaggle.com/datasets/soumicksarker/multiple-cameras-fall-dataset
- **Variables**: The dataset includes features such as `chute`, `cam`, `start`, `end`, `label` (target variable for fall/no fall).
- **Size**: The dataset comes from 23 different scenarios, where each video had lengths between 6 to 30 seconds. Where 40 fps at 480x720x3 frames were obtained. 

The dataset was obtained by retrieving the frames using `start` and `end` for each of the rows of the metadata dataframe. Depending as well weather it was fall detection (CNNs), and most importantly, fall prediction (CNN-LSTM). Data preprocessing was made.

For the CNNs, the optical flow gradietns using Farnerback method was done and utilized in order to capture movement when a fall was occuring. In the case of the CNN-LSTM approach, 3D pose estimation was done and annotated in the images; which then was further passed in sequences for a predictive horizon of 600ms 

### Dataset Preparations

Download the dataset from Kaggle link. Once inside, create a `data` folder where you would contain the `MultiCamFall` data. The files it should contain are:
1. dataset folder
2. `data_tuple3.csv` dataframe with metadata
3. `technicalReport.pdf` containing all technical recording information regarding the MultiCamera Fall Dataset.

#### Models: Used
- **VGG16-Finetuned**: This CNN baseline model was retrieved with Imagenet weights, and then finetuned for our binary classification task.
- **Optimized MyLeNet**: Inspired on the MyLeNet CNN, we trained a model for the binary classification task. We added bacthnormalization, and much more activation mappings to see if it behaved better than the VGG16.
- **CNN-LSTM**: Used to model to predict a fall beased on a predictive horizon og 600ms, The first 500ms corresponded to event before the fall, and the last 100ms for was overlapping window containing the fall event.

#### Project Steps:
1. **Data Cleaning/Preparation CNN**: Since the CNN was based on frames, as well as using optical flow gradients, an entire folder with its own data cleaning and preparation was made.
2. **Data Cleaning/Preparation CNN-LSTM**: Since the CNN-LSTM model required sequences of frames to predict a fall, as well as 3D posing annotation of these sequences. An entire folder with its own data cleaning and preparation was made.
3. **EDA CNN Optical Flow Gradients**: The data was visualized for CNN optical flow gradients between different cameras and different chutes. It was found that the distance of the camera, as well as the resolution, would have a huge impact weather the optical flow gradient that was obtained was correct.
3. **EDA CNN-LSTM Sequences**: Before even training, it was required to do an EDA of the proper sequences with 3D posing annotations.
4. **CNN Model Training and Analysis**: The models were evaluated based on F1 score, Accuracies, Precisions, Recalls, and F1 Scores. OUR GOAL too was to never predict False Negatives (A model must be always safe to predict a fall).
5. **CNN-LSTM Model Training and Analysis**: Same as the CNN metrics, with the added caviat of looking across time too how it behaved.
5. **Conclusion & Recommendations**: The CNNs had virtually the same F1 scores = 0.66. These F1 scores were also lower against the CNN-LSTM model of 0.78. Its also key to show that CNN-LSTM Recalls was 0.94 vs 0.66 of the CNNs. Illustrating the higher capability of CNN-LSTM model complexity. 

Laslty, the task of the CNN-LSTM model was WAY more complicated; to predict a fall in the future. These results showcase promising research on the field to predict a fall. Considerations such as:
1. Retrieving weights from CNNs that were pretrained on human motion (UCF-101 dataset), and reused in the CNN-LSTM.
2. More data for feeding CNN-LSTM in the closed stochastic environment of a hospital.
3. Consideration of predictive horizon based models.
#### License
This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

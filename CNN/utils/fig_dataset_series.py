'''
Date: 2025-02-11
Authors: Juan Pablo Triana Martinez
'''
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class FigDatasetSeries(Dataset):
    '''
    Class that would inherit from the Dataset class
    in order to read the Fall/No Fall .npy files
    of Optical flow gradients.

    This would load the files in a series fashion,
    since we are doing 30 frames per second. The shape
    of each item would be (30, 2, 224, 224) and the label
    (30, 1)
    '''

    def __init__(self, dataframe: pd.DataFrame, data_path: str,
            device:str) -> None:
        '''
        Parameters:
            dataframe(pd.Dataframe): Contains all metadata information
            required in order to read the data.
            data_path(str): The string path to the images folder, where
            the .npy files are stored
        '''
        super().__init__()

        #Assign the variables
        self.dataframe = dataframe
        self.data_path = data_path
        self.device = device
    
    def __len__(self):
        '''
        Instace method from the abstract class Dataset from torch,
        would return the total number of series in the dataset
        '''
        return len(self.dataframe)

    def __getitem__(self, index:int) -> tuple[np.ndarray, np.ndarray]:
        '''
        Instance method from the abstract class Dataset from torch,
        that would do the following:
        1.) Obtain the important information from the dataframe.
        2.) Get the filename of the Optical flow.
        3.) Read the Optical flow .npy file.
        4.) Return the Original Images
        5.) Return the Optical Flows Images.
        6.) Return the labels
        '''
        #Obtain the row of interest
        row = self.dataframe.iloc[index]

        #Obtain the important information
        chute = int(row['chute'])
        cam = int(row['cam'])
        start = int(row['start'])
        end = int(row['end'])
        label = int(row["label"])

        #Name of the video
        frame_name = f"chute{chute:02d}_cam{cam}_frames_{start}_{end}_"

        # If label = 1:
        if label == 1:
            # Access the fall folder
            fall_path = os.path.join(self.data_path, "fall")

            #Iterate files with .npy for ofs
            ofs_retrieved = []
            orig_images = []

            for file in os.listdir(fall_path):
                if file.startswith(frame_name) and file.endswith(".npy"):
                    # Get the file path
                    file_path = os.path.join(fall_path, file)

                    # Read the .npy file
                    flow = np.load(file_path)
                    ofs_retrieved.append(flow)
                elif file.startswith(frame_name) and file.endswith(".jpg"):
                    # Get the file path
                    file_path = os.path.join(fall_path, file)

                    # Read the .jpg file
                    image = np.array(Image.open(file_path))
                    orig_images.append(image)
                else:
                    continue
            
            #Cast it into a numpy array
            ofs_retrieved = np.array(ofs_retrieved)
            orig_images = np.array(orig_images)

            #Cast the ofs_retrieved to a tensor
            ofs_retrieved = torch.tensor(ofs_retrieved, dtype=torch.float32)
            orig_images = torch.tensor(orig_images, dtype=torch.float32)

            #Obtain the label
            label = torch.ones(ofs_retrieved.shape[0], dtype = torch.float32)

        else:
            # Access the no_fall folder
            no_fall_path = os.path.join(self.data_path, "no_fall")

            #Iterate files with .npy for ofs
            ofs_retrieved = []
            orig_images = []
            for file in os.listdir(no_fall_path):
                if file.startswith(frame_name) and file.endswith(".npy"):
                    # Get the file path
                    file_path = os.path.join(no_fall_path, file)

                    # Read the .npy file
                    flow = np.load(file_path)
                    ofs_retrieved.append(flow)
                elif file.startswith(frame_name) and file.endswith(".jpg"):
                    # Get the file path
                    file_path = os.path.join(no_fall_path, file)

                    # Read the .jpg file
                    image = np.array(Image.open(file_path))
                    orig_images.append(image)
                else:
                    continue

            #Cast it into a numpy array
            ofs_retrieved = np.array(ofs_retrieved)
            orig_images = np.array(orig_images)

            #Cast the ofs_retrieved to a tensor
            ofs_retrieved = torch.tensor(ofs_retrieved, dtype=torch.float32)
            orig_images = torch.tensor(orig_images, dtype=torch.float32)

            #Obtain the label
            label = torch.zeros(ofs_retrieved.shape[0], dtype = torch.float32)
        return orig_images, ofs_retrieved, label
'''
Date: 2025-02-11
Authors: Juan Pablo Triana Martinez
'''
import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class FigDatasetFrame(Dataset):
    '''
    Class that would inherit from the Dataset class
    in order to read the Fall/No Fall .npy files
    of Optical flow gradients.

    This would load the files in a frame fashion,

    '''

    def __init__(self, dataframe: pd.DataFrame, data_path: str,
            device:str, frames_per_event:int = 30) -> None:
        '''
        Parameters:
            dataframe(pd.Dataframe): Contains all metadata information
            required in order to read the data.
            data_path(str): The string path to the images folder, where
            the .npy files are stored
            frames_per_event(int): Number of frames per fall/No fall event
        '''
        super().__init__()

        #Assign the variables
        self.dataframe = dataframe
        self.data_path = data_path
        self.device = device
        self.frames_per_event = frames_per_event
    
    def __len__(self):
        '''
        Instace method from the abstract class Dataset from torch,
        would return the total number of entries in the dataset
        '''
        return len(self.dataframe) * self.frames_per_event

    def __getitem__(self, index:int) -> tuple[np.ndarray, np.ndarray]:
        '''
        Instance method from the abstract class Dataset from torch,
        that would do the following:
        1.) Obtain the important information from the dataframe.
        2.) Get the filename of the Optical flow.
        3.) Read the Optical flow .npy file.
        4.) Read the .jpg file for original image
        5.) Return the original image, Optical flow, and the labels.
        '''
        # Determine which sequence (video segment) the index belongs to
        seq_idx = index // self.frames_per_event  # Find the sequence index
        frame_idx = index % self.frames_per_event  # Find the specific frame index

        #Obtain the row of interest
        row = self.dataframe.iloc[seq_idx]

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
            ofs_retrieved = []
            orig_images = []
            for file in os.listdir(fall_path):
                #Iterate files with .npy
                if file.startswith(frame_name) and file.endswith(".npy"):
                    # Get the file path
                    file_path = os.path.join(fall_path, file)

                    # Read the .npy file
                    flow = np.load(file_path)
                    ofs_retrieved.append(flow)
                elif file.startswith(frame_name) and file.endswith(".jpg"):
                    # Get the file path
                    file_path = os.path.join(fall_path, file)

                    #Read the jpg file
                    image = Image.open(file_path)
                    orig_images.append(image)
                else:
                    continue
            #Cast it into a numpy array
            ofs_retrieved = np.array(ofs_retrieved)
            orig_images = np.array(orig_images)

            # Select the image at the specific frame index
            of_retrieved = ofs_retrieved[frame_idx]
            orig_image = orig_images[frame_idx]

            #Cast the ofs_retrieved to a tensor
            of_retrieved = torch.tensor(of_retrieved, dtype=torch.float32)
            orig_image = torch.tensor(orig_image, dtype=torch.float32)

            #Obtain the label
            label = torch.ones(1, dtype = torch.float32)

        else:
            # Access the no_fall folder
            no_fall_path = os.path.join(self.data_path, "no_fall")
            ofs_retrieved = []
            orig_images = []
            #Iterate files with .npy
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

                    #Read the jpg file
                    image = Image.open(file_path)
                    orig_images.append(image)
                else:
                    continue
            #Cast it into a numpy array
            ofs_retrieved = np.array(ofs_retrieved)
            orig_images = np.array(orig_images)

            # Select the image at the specific frame index
            of_retrieved = ofs_retrieved[frame_idx]
            orig_image = orig_images[frame_idx]

            #Cast the ofs_retrieved to a tensor
            of_retrieved = torch.tensor(of_retrieved, dtype=torch.float32)
            orig_image = torch.tensor(orig_image, dtype=torch.float32)

            #Obtain the label
            label = torch.zeros(1, dtype = torch.float32)
        return orig_image, of_retrieved, label
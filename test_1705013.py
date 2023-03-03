import argparse
import os
import cv2
import numpy as np
import pandas as pd

from train_1705013 import get_model


if __name__ == '__main__':
    # Parse the command line argument for the folder path
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_path', type=str, help='Path to the folder containing the images to be classified')
    args = parser.parse_args()
    
    # Load the weights and bias from model from the pickle file
    model = get_model('1705013_model.pkl')
    predictions = []
    target_size = (28, 28)

    # Loop through all the images in the folder
    image_names = []
    images = []
    # images = [f for f in os.listdir(args.folder_path) if f.endswith('.png')]
    for image_name in os.listdir(args.folder_path):
        if image_name.endswith('.png'):
            image_names.append(image_name)
            image_path = os.path.join(args.folder_path, image_name)
            img = cv2.imread(image_path, 0)              
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            images.append(img)
    #normalizing the images
    mean = np.mean(images)
    std = np.std(images)
    images = (images - mean) / std

    # Convert the list of images to a numpy array
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)  # Add a channel dimension (3D to 4D)

    test_scores = model.forward(images)
    y_pred = np.argmax(test_scores, axis=1)  
    for i in range(len(y_pred)):
        predictions.append([image_names[i], str(y_pred[i])])
        
    # Create a DataFrame from the list of predictions
    df = pd.DataFrame(predictions, columns=['FileName', 'Digit'])

    # Save the DataFrame to a CSV file
    csv_path = os.path.join(args.folder_path, '1705013_prediction.csv')
    df.to_csv(csv_path, index=False)


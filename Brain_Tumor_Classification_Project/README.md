
# Brain Tumor Classification Project

## Overview
This project aimed to classify brain tumors using convolutional neural networks (CNNs). We used image data from a labeled dataset containing brain MRI scans annotated with tumor types. 
By training a deep learning model, we could classify these MRI images into different categories. This project utilized the COCO (Common Objects in Context) dataset format for annotations, and we extracted this data to structure it for use in our model.

## Project Structure
The project involved several main steps, each playing a key role in preparing and utilizing the data for training, validating, and testing our model. Here’s an overview of each step we took:

1. **Dataset Preparation**
   - The dataset for this project included MRI images of brain scans that had been annotated for various types of brain tumors.
   - Annotations were stored in JSON files formatted in COCO format. The COCO format is widely used in computer vision and includes information about the images, annotations, and categories:
     - **Images**: Contained metadata about the images, such as image IDs and file names.
     - **Annotations**: Described the objects in the images, including category IDs and image IDs to link them back to the images.
     - **Categories**: Listed the possible tumor classes with their corresponding IDs and labels.
   
2. **Using COCO JSON Files**
   - We first read the JSON annotation files for each data split: training, validation, and testing.
   - By extracting details from the JSON files, we mapped each image to its category labels. This helped us understand which class each MRI scan belonged to.
   - Next, we listed the different category IDs and their respective names to identify all possible classes within our dataset.

3. **Data Exploration and Extraction**
   - Using the information from the COCO JSON files, we explored the categories present in the data. This provided a clear idea of what each category ID represented, such as different types of tumors.
   - We then extracted essential data from the JSON files to structure the information into a tabular format using Pandas DataFrames. Each entry in this table included the image file name, the tumor category ID, and the corresponding category name. This format made it easy to manage and utilize the data for training our machine learning model.

4. **Data Loading and Preprocessing**
   - We loaded the MRI images and their corresponding labels into memory. Preprocessing included resizing the images to a uniform shape to fit the input dimensions of the model.
   - Data augmentation was applied to increase the diversity of training samples, making the model more robust. This involved transformations such as random rotations, flips, and brightness adjustments.

5. **Model Building**
   - We constructed a Convolutional Neural Network (CNN) to classify the brain MRI scans. The CNN architecture included several convolutional layers for feature extraction, followed by fully connected layers for classification.
   - We compiled the model using an appropriate optimizer and loss function. In this case, we used **Sparse Categorical Crossentropy** for multi-class classification and **Adam** as the optimizer.
   - After defining the model, we reviewed its summary to ensure that all layers were correctly set up and that the output matched the number of tumor categories.

6. **Model Training**
   - The model was trained on the prepared dataset over **30 epochs** with a **batch size of 32**, improving training efficiency.
   - During training, we monitored the model's performance using accuracy and validation metrics to see how well it generalized to unseen data.
   - The validation dataset helped tune the model to prevent overfitting while ensuring high accuracy.

7. **Evaluation and Testing**
   - After training, the model was evaluated on the test set to assess its final performance. We used metrics such as accuracy, precision, recall, and F1-score to understand the model’s effectiveness.
   - A confusion matrix was also generated to visualize the performance across each class. This helped us see where the model might be confusing one class for another and provided insight into potential areas for improvement.

8. **Results and Observations**
   - The results showed that the model achieved a very high or near-perfect training accuracy of around **100%** and a very high validation accuracy of **99.30%**. The validation loss fluctuated based on the number of epochs the model was trained for.
   - From the visualization, specifically the confusion matrix, we were able to see how the model performed across different classes. 
   - The model did not overfit or underfit, as the training accuracy and test accuracy were very close. The confusion matrix served as a backup to assess whether the model overfit or not, in combination with other metrics.

## Conclusion
This project showcased the process of creating a CNN-based model for brain tumor classification from MRI scans. By using the COCO format for data annotation, we gained experience handling structured JSON data in machine learning tasks. The model’s high accuracy and consistency across different data splits suggested that it effectively distinguished between different tumor types, which could be beneficial in medical diagnostics.

## Requirements
To reproduce this project, you’ll need:
- Python 3.x
- Jupyter Notebook or a Python environment like Google Colab
- Libraries: TensorFlow, Keras, Pandas, NumPy, Matplotlib, Seaborn, and `pycocotools` for handling COCO format.

## Additional Notes
The full code for this project, including data loading, preprocessing, model training, and evaluation, can be found in the Jupyter Notebook file accompanying this README file.
Each cell in the notebook contains comments for better understanding and ease of replication.
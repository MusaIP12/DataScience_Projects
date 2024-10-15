
# COVID-19 Classification Using Audio Features

## Project Overview
This project focused on classifying COVID-19, PNEUMONIA, and NORMAL (healthy) cases based on audio features extracted from lung sound recordings. 
The primary goal is to use **machine learning together with audio data**to differentiate between these classes, aiding in the rapid identification and diagnosis of respiratory conditions.

### Background
Audio-based analysis for disease detection has gained traction as an innovative approach to healthcare. By analyzing patterns in lung sounds, this project aims to detect abnormalities associated with COVID-19, PNEUMONIA, and normal lung conditions. 
Machine learning models are trained to classify these recordings based on distinct audio features, making the process efficient and automated.

---

**Dataset** : Dataset was downloaded from kaggle- https://www.kaggle.com/datasets/alsaniipe/chest-x-ray-image

## Methods and Techniques

1. **Data Preparation**
   - **Audio Conversion**: The data preparation process began with a unique approach: converting the COVID-19 dataset images into audio files. 
   To achieve this, each image representing lung conditions was converted to audio by mapping pixel intensities to sound frequencies. 
   This transformation allowed for the analysis of pixel-based features in the auditory domain, making it possible to apply machine learning techniques for sound-based classification.
   
   - **Data Scaling**: The audio features were scaled using `StandardScaler` to ensure that all features contribute equally to the classification. This is important when using distance-based algorithms like KNN and gradient-based algorithms like XGBoost.
   
   - **Train-Test Split**: To evaluate the models, the data was split into training and test sets (80/20) with `train_test_split`. **Stratified sampling** was used to ensure balanced distribution across classes, maintaining the class ratio in both the training and testing sets.

2. **Audio Features and Feature Extraction**
Audio features are distinct, measurable characteristics of audio files, capturing information about the frequency, rhythm, timbre, and amplitude of sounds. By extracting these features, we transform each audio file into a numerical representation that can be fed into machine learning models. For this project, several key audio features were extracted using the **`librosa`** library:

   - **Mel-Frequency Cepstral Coefficients (MFCCs)**: These coefficients provide a representation of the short-term power spectrum of the audio signal. MFCCs are often used in audio analysis because they capture the perceptually important aspects of sound, including pitch and tone. They are calculated by applying a Fourier transform on overlapping frames of the audio file, then mapping these frequencies onto the Mel scale.
   
   - **Zero-Crossing Rate (ZCR)**: This measures how often the audio signal crosses zero amplitude within a given frame, reflecting the noisiness or texture of the sound. ZCR is particularly useful for distinguishing between sounds with different timbres and is commonly used in music and speech classification.
   
   - **Spectral Centroid**: This feature represents the "center of mass" of the spectrum and indicates where most of the spectral energy is concentrated. It provides insight into the brightness of a sound—higher spectral centroids correspond to brighter sounds, while lower values are associated with darker sounds.
   
   - **Spectral Bandwidth**: Spectral bandwidth measures the range of frequencies in the audio signal around the spectral centroid. It helps in identifying whether the sound is smooth (narrow bandwidth) or complex (broad bandwidth).
   
   - **Spectral Contrast**: This feature represents the difference in amplitude between peaks and valleys in the spectrum, which gives information about the harmonic content of the sound. Spectral contrast is particularly useful for distinguishing between sounds with similar overall energy but different frequency distributions.
   
   - **Spectral Roll-off**: Spectral roll-off is the frequency below which a certain percentage (e.g., 85%) of the total spectral energy is contained. This feature can help identify sounds that have strong low-frequency components versus those that are more high-frequency-oriented.
   
   - **Root Mean Square Energy (RMSE)**: RMSE is a measure of the loudness or energy of the audio signal over time. It is useful for identifying variations in volume within a recording, capturing details such as breaths, coughs, or other fluctuations in sound intensity.

   The extracted features were then concatenated into a single array for each audio recording, providing a comprehensive numerical representation of the sound profile for classification.

3. **Model Selection**
Several machine learning classifiers were selected for training and testing due to their suitability for multi-class classification and performance:

   - **Random Forest**
   - **Decision Tree**
   - **K-Nearest Neighbors (KNN)**
   - **LightGBM**
   - **XGBoost**

4. **Model Evaluation**
   Each model was evaluated using several metrics:
   - **Accuracy**: The percentage of correctly classified samples.
   - **Precision, Recall, and F1-Score**: Detailed in the classification report, these metrics evaluate the models' performance across each class.
   - **Confusion Matrix**: Provides a breakdown of predictions by class, aiding in the visualization of model performance on different categories.
   - **ROC AUC**: The ROC AUC score is used as a measure of separability. Higher values indicate that the model is better at distinguishing between classes.

---

## Results

### Summary of Classifier Performance
The results of the five classifiers are summarized as follows:

- **Random Forest**: Achieved an accuracy of 86.32% and an ROC AUC of 0.9534. This model showed strong performance in detecting PNEUMONIA but had lower recall for COVID-19.
  
- **Decision Tree**: Scored an accuracy of 78.94% and an ROC AUC of 0.7652. While it provided quick training and testing, the model underperformed relative to the others due to its lack of complexity and depth.

- **K-Nearest Neighbors (KNN)**: Achieved an accuracy of 80.57% and an ROC AUC of 0.8516. It performed well with PNEUMONIA detection, but it had slower testing times due to its reliance on distance calculations.

- **LightGBM**: This model achieved the highest accuracy of 89.43% and an ROC AUC of 0.9671. LightGBM exhibited high recall across all classes and balanced precision-recall, making it one of the best performers.

- **XGBoost**: Very close to LightGBM, XGBoost achieved an accuracy of 89.20% and an ROC AUC of 0.9652. It showed a balanced performance across classes with fast training times.

### Key Observations
- **Best Performer**: LightGBM slightly outperformed the other models, making it the highest performer for this classification task.
- **Class Imbalance**: The COVID-19 class, being the smallest in the dataset, showed a tendency for lower recall across models. Addressing this imbalance could improve the model's sensitivity to this class.
- **Gradient Boosting Models**: Both LightGBM and XGBoost demonstrated the best balance of speed and accuracy, highlighting their suitability for high-dimensional feature sets in audio data classification.

---

## Conclusions and Future Work
This project demonstrates that audio features can be used effectively to classify COVID-19, PNEUMONIA, and NORMAL cases, with LightGBM and XGBoost being the most successful model in terms of both accuracy and computational efficiency.
**The project has also created a new audio dataset which can be used for future work purposes.**

### Conclusions
- **Feasibility**: The results confirm the feasibility of using audio-based machine learning for respiratory disease classification, particularly in distinguishing between COVID-19 and other conditions.
- **Performance**: Gradient boosting algorithms (LightGBM and XGBoost) provide the best performance in terms of accuracy and ROC AUC, making them promising candidates for deployment in real-world applications.
  
### Future Improvements
To enhance the model further:
- **Class Balancing**: Techniques such as SMOTE (Synthetic Minority Oversampling Technique) could be used to address the imbalance in the dataset, particularly to improve the detection of COVID-19 cases.
- **Additional Features**: Further feature engineering or inclusion of other audio features (e.g., Chroma features, Spectral Contrast) could provide additional information for the classifiers.
- **Ensemble Techniques**: Combining top models in a voting ensemble could harness their strengths and improve classification stability.
- **Optimization**: Optimization techniques can applied to this technique to improve the performance of models. e.g Using metaheuristic algorithms to optimize feature selection process
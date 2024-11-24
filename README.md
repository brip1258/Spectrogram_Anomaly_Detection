# ROAD_anomaly_detection
## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Report](#report)

## Installation
1. Clone the respository:
```bash
git clone https://github.com/brip1258/Spectrogram_Anomaly_Detection.git
```
2. Navigate to the project directory: 
```bash
cd Spectrogram_Anomaly_Detection
```
3. Create and activate a virtual environment (Optional but recommended):
```bash
python3 -m venv env
source env/bin/activate
```
4. Install the required dependencies:
```bash
pip install -r requirements.txt
```
## Usage
To run the script, use the below command:
```bash
python main.py 
```

## Report
### Data Exploration:
For my data exploration (EDA) with the ROAD_dataset.h5 file, I started by exploring the structure of the file. I created functions that would evaluate each group and its contents. There were 10 anomaly groupings with imbalanced sample sizes, ranging from 56 (Oscillating Tiles) to 550 (Galactic Plane). Additionally, there were observations categorized as "Normal" and "All non-characterized effects." Based on this description, I believe, the data can have "Normal" occurrences outside of the already identified anomalies, but there could be occurrences of anomalies within data set itself. This opens the door to supervised and unsupervised approach that can be taken into consideration. 
    
I explored the data characteristics using pair plots to visualize relationships between features. To identify patterns, I plotted all observations as well as a single row of observation, within each grouping together to examine group-level trends and analyzed individual rows to compare specific observations. While I considered creating additional features, based on my understanding raw data had already been reduced in dimensionality during its creation using auto-correlation. Given this, further aggregation could misrepresent the data and obscure meaningful patterns, emphasizing the need to preserve the original structure of the observations.

I also evaluated other characteristics such as minimum, maximum, average, and standard deviation to understand the distribution of the groupings within the data set. I confirmed that there were no null values within the dataset. To further identify patterns I created heat maps of each grouping. The first set of heat maps represent a visualization of the spectrogram data from a single i-th indexed observation. This shows key features, like intensity noise patterns and repetitive frequency patterns like what is shown in First Order Noise and the Solar Flares occurrences have similar looking heat maps. This could potentially lead to misclassifications. I also, took a look at the heatmap for frequency bands which would be a good representation of the magnitude of frequency. This in a similar manner as the previous heat map can show us differences and similarities amongst the varying grouped data of frequency band characteristics. It is also evident that there multiple classes that represent on observation, which also leads to overlap in the data. This means that we can also conduct multi-label classification as well. For the purposes of sampling we will only use labels that have one classification to ensure we can confidently classify these anomalies. Another finding was that the train data has two labels, and empty string and a string of 1. Though I couldn't find direct definition of which label represents normal and the other potentially being anomalous, you can see that the graphical representation of the empty string values in the EDA, differ highly from the anomalous occurrences, therefore will be considered normal.


### Model
Prior to modeling I had to preprocess the training data. As mentioned in the above there were multi label representations within the data but a majority of data had single classifications of either 1 of the 9 anomalies (oscillating tile, first order high noise, first order data loss, third order data loss, lightning, rfi ionosphere reflect, galatic plane, source in sidelobes, or solar storm) or the classification of normal. Therefore for the training data I utilized a subset where the observations only had a single classification. For normal data, I utilized all values within the train that had no labels as normal. Due to the imbalance in data, I wanted as close to equal representation of each category as possible to ensure that we didn't have a significant amount of bias introduced within our training data. When training the model the total sample of 500 observations where each label represented 10% of the sample. My input values were the spectrogram data, represented in 4-D (time, frequency,polarisation, and station) while my output was the classification of normal or one of the define anomalies. 

I started with a baseline model, utilizing random forest to understand where we can improve from. From there I decided to utilize a convolutional neural network (CNN). A CNN was chosen because this is a multi dimensional dataset. CNN's are ideal in extracting features from these types of data. Also, with this type of data and what can be seen through the EDA there is quite a bit of noise, CNN's are ideal in extracting the important features and ignoring irrelavant details from the noise to make accurate predictions. Based on this training my CNN had an Accuracy of 96, a precision score of 97, and a  recall score of 96. This is fairly high just for an initial training meaning that we maybe over fitting our model. It would be ideal to adjust the sample size to ensure that our model isn't overfitting. This would be difficult because of the aforementioned imbalance in data. 

For me to improve the model overall we would need to obtain more data or we could create augments/synthetic data. For me to determine if this is a 'good' model or not, would be dependent on the context of which we are applying this model. If we were to make further improvements I would create expand the sample size as much as possible including augmented/synthetic data. I would modify the model architecture, utilize different epoch values, introduce band frequency. Another avenue is to take is perhaps utilizing a 2 classification system of normal and not normal, the performance could be significanly better and you may be able to utilize a lighter weight model. Though my development is a good start there is a significant number of avenues that can be taken to evaulate this data set, develop the model, and implement it to scale. 
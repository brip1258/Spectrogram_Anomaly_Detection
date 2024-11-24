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
Prior to modeling, I had to preprocess the training data. The data set included multi-labeled observations, but the majority of the observations had single classifications of 1 of the 9 anomalies (oscillating tile, first order high noise, first order data loss, third order data loss, lightning, rfi ionosphere reflect, galactic plane, source in sidelobes, or solar storm) or the classification of normal. Therefore for the training data I utilized a subset where the observations only had a single classification. Normal data was identified as observations within the train data that had no labels. To address the class imbalance, I wanted as close to equal representation of each category as possible minimize bias introduced within our training data. The model was trained using a total sample of 500 observations where each category represented 10% of the sample. The input values were the spectrogram data, represented in 4-D (time, frequency, polarisation, and station) while my output was the classification of normal or one of the define anomalies. 

I used Random Forest as a baseline model to evaluate where improvements could be made. From there I decided to utilize a convolutional neural network (CNN). A CNN was chosen because this is a multi dimensional dataset. CNNs are well-suited for multidimensional datasets like this, as they are highly effective at extracting meaningful features from complex data structures. Additionally, the exploratory data analysis (EDA) revealed significant noise in the dataset. CNNs excel at isolating important features while ignoring irrelevant noise, leading to more accurate predictions.

CNN Results:
- Accuracy of 96%
- Precision of 97%
- Recall of 96% 

These results are promising, but the high scores suggest the potential for overfitting. This could be from the limited sample size that was used. In order to address this I could increase the sample size, which is conditional due to the imbalance int he original data set.

For future improvements we can take steps by obtaining more data or I could create augments/synthetic data. I could also, modify the model architecture, utilize different epoch values, utilize a different architecture, or adjust different hyper parameters. Another aspect would to be to introduce another feature that was represented in the data, band frequency. I could also simplify the problem if allowed to a binary classification system of normal and not normal, the performance could be significantly better and you may be able to utilize a lighter weight model. 

For me to determine if this is a 'good' model or not, would be dependent on the context of which we are applying this model. Though my development is a good start there is a significant number of avenues that can be taken to evaluate this data set, develop the model, and implement it to scale. 
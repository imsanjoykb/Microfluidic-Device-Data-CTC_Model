## Classification of Circulating Tumor Cells Using Machine Learning on Microfluidic Trajectory Data
<hr>

![alt text](model_architecture/readme_banner.png "GRU Architecture")

[![Python](https://img.shields.io/badge/Python-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Numpy](https://img.shields.io/badge/Numpy-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c.svg?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-3776ab.svg?logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MIT](https://img.shields.io/badge/license-MIT-5eba00.svg)](https://github.com/imsanjoykb/Text-Generation/LICENCE.txt)
[![DOI](https://img.shields.io/badge/DOI-10.1000%2Fxyz123-blue)](https://doi.org/10.1000/xyz123)
[![GitHub stars](https://badgen.net/github/stars/imsanjoykb/vMDpcDI-CTC_Modeling)](https://github.com/imsanjoykb/vMDpcDI-CTC_Modeling/stargazers)

<hr>

## Abstract
>This study investigates the use of machine learning techniques to classify circulating tumor cells (CTCs) based on their movement patterns within a hyperuniform micropost microfluidic device. Utilizing cell-based modeling, a synthetic dataset was created to simulate the behavior of CTCs in blood flow. Three machine learning models were employed to analyze the trajectory data: a Convolutional Neural Network (CNN), a hybrid model combining CNNs and Long Short-Term Memory (LSTM) networks, and the eXtreme Gradient Boosting (XGBoost) algorithm. These models achieved an average classification accuracy of 80% in identifying distinct CTC phenotypes, demonstrating the potential of this method for early cancer detection.
<hr>

## Project Installation and Run

<b>Create the virtual environment</b>
```
pip install virtualenv
virtualenv venv 
venv\Scripts\activate  
```
<b>Install Dependencies</b>
```
pip3 install -r requirements.txt
```
<b>Run CNN Code/Notebook</b>
```
python CNN_Model.py
or,
CNN_Model.ipynb
```
<b>Run Hybrid Model Code/Notebook</b>
```
python Hybrid_Model.py
or,
Hybrid_Model.ipynb
```
<b>Run XGBoost Model Code/Notebook</b>
```
python XGBoost_Model.py
or,
XGBoost_Model.ipynb
```
<b>Run Final and Merge Model</b>
```
Final_Models.ipynb
```
<hr>




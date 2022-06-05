# GTZAN Music Genre Embeddings

The goal was to implement the API from [this](https://hearbenchmark.com/hear-api.html) challenge.

Two different variation are tried for this task. 
* MFCC Features
* LSTM

Code Structure:
* **classifier_utility.py**: Functions for rapidly training and testing LogisticRegression Models
* **create_split.py**: Utility to divide the dataset into splits of train, test and test_embed into proportions of 0.6, 0.2 and 0.2 respectively.
* **data_utils.py**: Functions to load and process audio files
* **embedding_api.py**: Implementation of the Hear 2021 API
* **models.py**: MFCC and LSTM Models used for generating embeddings
* **model-train-and-save.ipynb**: Code for training (LSTM model) and saving (both) models
* **results.ipynb**: Final results of the different embedding variants along with comments about potential causes for disparities.

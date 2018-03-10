# Exoplanet-Detection
This repository contains the code for Exoplanet Detection using machine learning. We have used a novel approach of anomaly detection using variational autoencoders in order to detect anomalies(Exoplanets). As the dataset is heavily skewed toward non-exoplanets(Exoplanets are incredibly rare), we have trained the variational autoencoder only on non exoplanets. A large error in reconstruction indicates the presence of a possible exoplanet.
The full problem statement can be found [here](https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data)


## Dependencies

* Keras
* Pandas
* Tensorflow 
* Scipy 
* Matplotlib 
* Pickle
* sklearn

## Visualisations

Visualizations of the dataset can be found in the ipython notebook. 


### Training the Autoencoder
In order to train the neural network, first run the data preprocessing script
```
python3 data_preprocessing.py
```
To run the tensorflow version, run

```
sudo python3 train_tf.py
```

To run the Keras version, run 
```
sudo python3 train_keras.py
```
Checkpoints of trained weights can be found in the checkpoints folder.


### Testing 

To test the autoencoder paste the test data in data directory and run
```
python3 test.py
```

### Sources

Credits go to [MuonNeutrino](https://www.kaggle.com/muonneutrino/exoplanet-data-visualization-and-exploration) for the visualizations and Abhinau Kumar for the keras code.


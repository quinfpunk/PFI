# PFI
Using Perceptive field to interpret CNN

## Project foundation
- A CNN trained on [this dataset](https://www.kaggle.com/datasets/oleksandershevchenko/ship-classification-dataset) 
- Gather the perceptive field
- Visualize the per neurons filters

## Usage
When training the CNN you should have the dataset named 'ships_dataset' and located in the same directory as the training notebook and utils.py

To get GradCam images, first in the remove_useless_neurons.py file change the target_class in the main, then execute the script:
```
python remove_useless_neurons.py
```

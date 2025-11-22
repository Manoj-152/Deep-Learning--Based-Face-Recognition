# Face Recognition using Deep Learning
This project implements a deep learning-based face recognition system using a ResNet18 backbone to extract discriminative facial feature representations. By computing cosine similarity between the generated feature vectors, the model determines whether two input images belong to the same person. The overall workflow is designed to be simple, efficient, and easy to use, allowing reliable face matching through straightforward training, evaluation, and analysis.

<p align="center">
  <img src="Examples/Dwayne-Johnson/Dwayne_Johnson_Compare_Result.png" width="300">
  <img src="Examples/Shah-Rukh-Khan/Shah-Rukh-Khan_Compare_Result.png" width="300">
</p>

*Refer to Examples folder to see some face comparisons done on celebrities*

*best_ckpt.pth contains the trained model weights*

## Requirements (Libraries)

* torch 1.7.0 
* torchvision 0.8.0
* opencv 3.2.0
* numpy 1.18.5
* matplotlib 3.3.2
* PIL 7.2.0
* tqdm 4.54.0

## About Files

* **train.py** — Train the model.
  - 500 epochs
  - Estimated time for completion: 2 hours
* **analysis.py** — Determine whether two face images match or mismatch.
  - Accepts paths of two pictures as input using argparse. (reference photo and selfie photo)
  - Example for running the code: `python3 analysis.py path_1 path_2`
* **evaluate.py** — Compute match and mismatch accuracies on any external dataset.
  - Accepts path of the dataset on which the evaluation must be done.
  - Example for running the code: `python3 evaluate.py trainset`
* **threshold_experimentation.py** — Used to estimate the optimal similarity threshold value.
* **resnet.py** — Contains the ResNet model class used to build the ResNet18 architecture.
* **dataloader.py** — Contains the `FaceDataset` class used to build the train and validation dataloaders.
  
## Accuracies on final model (best.ckpt) 
Accuracy for predicting match between faces : 84 %

Accuracy for predicting mis-match between faces : 81 %

(Calculated by using evaluate.py on the validation set)
  
## A Description of the code

* The ResNet18 architecture is used here to generate feature vectors for the faces. After generating the feature vectors, cosine similarity measure is used to find match and mis-match between images.
* The cosine similarity measure is printed here as the confidence score of matching between two face pictures. A threshold of 0.54 was set on the similarity measure for matching and mis-matching of faces.
* The threshold of 0.54 was chosen by analysing using the threshold_experimentation.py file, which could also be run to get a match accuracy vs threshold value and mis-match accuracy vs threshold value graph on the validation set. The graph results are also as follows:

<p align="center">
  <img src="validation_graph.png" width="350">
</p>

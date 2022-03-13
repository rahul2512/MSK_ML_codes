
<<<<<<< patch-1
# Machine Learning for Musculoskeletal Modeling of Upper Extremity

This file contains the code corresponding to the article "Machine Learning for Musculoskeletal Modeling of Upper Extremity".

A cluster was used to run cross-validation, so the code may require some changes to run on a laptop. Final model training can be easily done on a normal laptop. 

Test data for reproducing the results are provided here. The full training dataset can be obtained on request to vikranth.harthikotenagaraja@eng.ox.ac.uk

The codes are developed using Python 3 with [keras](https://keras.io) for ML.
=======
# MSK_ML_codes
# This file contains the code description used in the article: "Machine Learning for Musculoskeletal Modeling of Upper Extremity"
# The codes are developed in python3 using standard python modules and keras for ML. 

# The code here contains:
#a) Pipeline to run ML methods such NN, LM which can be easily extended for other methods. 
#b) We have used this pipeline to run cross-validation on cluster. Therefore, to run on laptop may require slight tweak in the code. 
#c) Final training and testing can be easily done on any laptop/computer. 
#d) Code for analysis and plotting is also provided. 

# Code description
#pytorch_utilities.py -- contains function for various models (Linear, Neural Network, ....) and generate a file with hyperparameters choices
#pytorch.py -- several functions to handle data, perform cross-validation, train model, forward pass, plot and analyse results
#analysis*py -- perform cross-validation 

# Test data for reproducing the results are provided here. The full training can be obtained on request to vikranth.harthikotenagaraja@eng.ox.ac.uk

 
>>>>>>> main

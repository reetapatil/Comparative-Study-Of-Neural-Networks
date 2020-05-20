
Requirements:

Tensorflow-GPU
Keras
Pandas
Joblib
Sklearn
Seaborn
Numpy
PIL

Data Preprocess:
Dataset given in this package does not need preprocessing. It has already been implemented.

Dataset: FER2013
Dataset link: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
Note: Dataset will be downloaded in CSV form. To convert data into images, execute read_data_execute_once.py once before running main_program.py of the project.

Code:
main_program.py : Driver program
cnn_models.py : cnn model definitions
cnn_and_svm.py : cnn+svm model definitions
read_data_execute_once.py: data preprocess program.
Utils.py : Utility program

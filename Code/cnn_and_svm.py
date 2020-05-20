import Utils
from Utils import load_model, feature_extractor_to_svm, svc, evaluate_svm_model_expression_error_rate
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn

#load the previously trained deep CNN model
h5filename = 'dense_cnn.h5'
model = load_model(h5filename)
model.summary()

#Extract features from last dense layer of CNN model
layer_name = 'dense_1'
train_features, train_labels = feature_extractor_to_svm(Utils.train_dir, model, layer_name=layer_name)
validation_features, validation_labels = feature_extractor_to_svm(Utils.validation_dir, model,
                                                                  layer_name=layer_name)
test_features, test_labels = feature_extractor_to_svm(Utils.test_dir, model, layer_name=layer_name)

#Train SVM using the extracted features
classifier, acc = svc(train_features, train_labels, validation_features, validation_labels)
print("Cnn and Svm (Accuracy %.2f%% )" % (acc * 100))

#Test SVM
score = classifier.score(test_features, test_labels)
print(score)
err = evaluate_svm_model_expression_error_rate(classifier, test_features, test_labels)
Utils.plt_expression(err, 'Individual expression error rate')
Y_pred = y_pred = classifier.predict(test_features)

#Confusion matrix
print('Confusion Matrix')
confusion = confusion_matrix(test_labels, y_pred)
print(confusion)
target_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
df_cm = pd.DataFrame(confusion, index=[i for i in target_names],
                     columns=[i for i in target_names])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.show()

#Classification report
print('Classification Report')
classification_report = classification_report(test_labels, y_pred, target_names=target_names)
print(classification_report)

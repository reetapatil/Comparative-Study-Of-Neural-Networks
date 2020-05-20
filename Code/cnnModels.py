from keras import layers
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.applications.vgg16 import VGG16
from sklearn.externals import joblib
from keras import models
import Utils
from Utils import load_model, feature_extractor_to_svm,svc,evaluate_svm_model_expression_error_rate, plt_expression



def basic_cnn():
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=1e-6),
                  metrics=['acc'])
    return model


def dense_cnn():
    # conv  block 1
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    # conv  block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    # conv  block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=1e-6),
                  metrics=['acc'])
    return model


def Lenet():
    # conv  block 1
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=1e-6),
                  metrics=['acc'])
    return model



def cnn_and_svm():
    #Load the dense model
    h5filename = 'dense_cnn.h5'
    model = load_model(h5filename)
    model.summary()
    layer_name = 'dense_1'
    train_features, train_labels = feature_extractor_to_svm(Utils.train_dir, model, layer_name=layer_name)
    validation_features, validation_labels = feature_extractor_to_svm(Utils.validation_dir, model,
                                                                      layer_name=layer_name)
    test_features, test_labels = feature_extractor_to_svm(Utils.test_dir, model, layer_name=layer_name)
    model_file = "cnn_and_svm.joblib"
    classifier, acc = svc(train_features, train_labels, validation_features, validation_labels)
    print("Cnn and Svm (Accuracy %.2f%% )" % (acc * 100))
    Utils.save_model(model, 'cnn_and_svm.h5')
    #classifier = joblib.load(model_file)
    score = classifier.score(test_features, test_labels)
    print(score)
    err= evaluate_svm_model_expression_error_rate(classifier,test_features, test_labels)
    Utils.plt_expression(err, 'Individual expression error rate')
    return classifier
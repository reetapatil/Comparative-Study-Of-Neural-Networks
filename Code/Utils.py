import os
from PIL import Image
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras import models
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sb



base_dir = os.getcwd()
dataset_dir = os.path.join(base_dir, 'dataset')
fer2013_csv = os.path.join(dataset_dir, 'fer2013.csv')
fer2013 = os.path.join(dataset_dir, 'fer2013')

train_dir = os.path.join(fer2013, 'Training')
validation_dir = os.path.join(fer2013, 'PublicTest')
test_dir = os.path.join(fer2013, 'PrivateTest')

Models = os.path.join(os.getcwd(), 'Models')

# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
digit2emotions = {'0': 'Angry', '1': 'Disgust', '2': 'Fear', '3': 'Happy',
                  '4': 'Sad', '5': 'Surprise', '6': 'Neutral'}


def plt_confusion_matrix(model):

    test_generator = image_data_generator(test_dir, shuffle=False, batch_size=20)
    Y_pred = model.predict_generator(test_generator, 180)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    confusion = confusion_matrix(test_generator.classes, y_pred)
    print(confusion)
    target_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    df_cm = pd.DataFrame(confusion, index=[i for i in target_names],
                         columns=[i for i in target_names])
    plt.figure(figsize=(10, 7))
    sb.heatmap(df_cm, annot=True)
    plt.show()

    print('Classification Report')
    classify_report = classification_report(test_generator.classes, y_pred, target_names=target_names)
    print(classify_report)

def pixel2image(pixels, dst_dir, fname, mode='L'):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    img_path = os.path.join(dst_dir, fname)
    im = Image.fromarray(pixels).convert(mode)
    im.save(img_path)


def csv2image():
    with open(fer2013_csv) as f:
        file = csv.reader(f)
        next(file)
        indices = 0
        for labels, pixels, tra_test in file:
            pixels = pixels.split()
            pixels = np.asarray(pixels, dtype=np.uint8).reshape(48, 48)
            subdir = os.path.join(fer2013, tra_test)
            emotions = digit2emotions[labels]
            dst_dir = os.path.join(subdir, emotions)
            pixel2image(pixels, dst_dir, '{}.{}.jpg'.format(emotions, indices))
            indices += 1
        print(indices)

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plt_acc_loss(history):

    acc = smooth_curve(history.history['acc'])
    val_acc = smooth_curve(history.history['val_acc'])
    loss = smooth_curve(history.history['loss'])
    val_loss = smooth_curve(history.history['val_loss'])

    epochs = range(1, len(acc) + 1)
    sb.set()
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.figure()
    sb.set()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()


def image_data_generator(data_dir,
                         data_augment=False,
                         batch_size=20,
                         target_size=(48, 48),
                         color_mode='grayscale',
                         class_mode='categorical',
                         shuffle=True):


    datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_directory(data_dir,
                                            target_size=target_size,
                                            color_mode=color_mode,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            class_mode=class_mode)
    return generator


def evaluate_model(model=None, filepath=None):

    if not model:
        assert(filepath)
        model = models.load_model(filepath)
    test_generator = image_data_generator(test_dir, batch_size=1, shuffle=False)

    nb_samples = len(test_generator)
    score = model.evaluate_generator(test_generator, steps=nb_samples)

    return score


def probas_to_classes(y_pred):

    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        return np.argmax(y_pred, axis=1)
    return np.array([1 if p > 0.5 else 0 for p in y_pred])


def evaluate_expression_error_rate(model):

    batch_size = 1
    test_generator = image_data_generator(test_dir, batch_size=batch_size)
    sample_count = len(test_generator)
    Y_test = np.zeros(shape=sample_count)
    Y_pred = np.zeros(shape=sample_count)

    i = 0
    for X, labels_batch in test_generator:

        pred = model.predict(X)
        Y_pred[i] = probas_to_classes(pred)
        # convert one=hot encode to index
        Y_test[i*batch_size: (i+1)*batch_size] = np.argmax(labels_batch, axis=1)
        i += 1
        if i * batch_size >= sample_count:
            break

    classes = len(set(Y_test))
    x_num, y_num = [0]*classes, [0]*classes
    for pred, test in zip(Y_pred, Y_test):
        y_num[int(test)] += 1
        if pred != test:
            x_num[int(test)] += 1

    err = [i/j for i, j in zip(x_num, y_num)]
    return err


def evaluate_svm_model_expression_error_rate(clf, test_features, test_labels):
    predict_test_label = clf.predict(test_features)
    classes = len(set(test_labels))
    class_total, error_labels = [0] * classes, [0] * classes
    for predict_label, test_label in zip(predict_test_label, test_labels):
        class_total[int(test_label)] += 1
        if predict_label != test_label:
            error_labels[int(test_label)] += 1

    return [error/total for error, total in zip(error_labels, class_total)]


def plt_expression(err, title):
    s = pd.Series(
        err,
        index=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    )

    sb.set()
    plt.title(title)
    plt.ylabel('error rate')
    plt.xlabel('emotions')

    my_colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c'] 

    s.plot(
        kind='bar',
        color=my_colors,
    )
    plt.show()


def save_model(model, filename):
    filepath = os.path.join(Models, filename)
    if filename.endswith('h5'):
        model.save(filepath)
    elif filename.endswith('json'):
        model.to_json(filepath)


def load_model(filename):
    Model = os.path.join(os.getcwd(), 'Models')
    filepath = os.path.join(Model, filename)
    model = None
    if filename.endswith('h5'):
        model = models.load_model(filepath)
    elif filename.endswith('hdf5'):
        model = models.load_model(filepath)
    elif filename.endswith('json'):
        with open('model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
            model = models.model_from_json(loaded_model_json)
    elif filename.endswith('pkl'):
        joblib.load(filepath)
    assert(model)
    return model


def feature_extractor_to_svm(directory, model, layer_name=None,
                             batch_size=20):

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    generator = image_data_generator(directory, batch_size=batch_size)
    sample_count = len(image_data_generator(directory, batch_size=1))
    print(os.path.split(directory)[-1], 'dataset: ', sample_count)

    features = np.zeros(shape=(sample_count, 1024))

    labels = np.zeros(shape=(sample_count))

    i = 0
    for inputs_batch, labels_batch in generator:
        intermediate_output = intermediate_layer_model.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = intermediate_output
        # one hot encoding to scalar enconde
        labels[i * batch_size: (i + 1) * batch_size] = np.argmax(labels_batch, axis=1)
        i += 1
        if i * batch_size >= sample_count:
            break
    np.reshape(features, (sample_count, 1024))
    return features, labels


def svc(traindata, trainlabel, testdata, testlabel):
    model_file = "cnn_and_svm_linear.joblib"
    print("Training SVM with rbf kernel funtion...")
    classifier = SVC(C=1.0, kernel="linear", cache_size=3000)
    classifier.fit(traindata, trainlabel)

    joblib.dump(classifier, model_file)
    print('dumped')
    pred_testlabel = classifier.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)

    return classifier, accuracy

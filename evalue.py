import os, glob
import matplotlib.pyplot as plt
import numpy as np
from park.Parking import Parking
from keras.models import load_model
import cv2
from sklearn import metrics
def keras_model(weights_path):
    model = load_model(weights_path)
    return model

if __name__ == '__main__':
    park = Parking()
    test_images_empty = [plt.imread(path) for path in glob.glob('train_data/train/empty/*.jpg')]
    test_images_occupied = [plt.imread(path) for path in glob.glob('train_data/train/occupied/*.jpg')]
    y_true = []
    y_pre = []
    weights_path = 'car1.h5'
    emptyLenth = len(test_images_empty)
    model = keras_model(weights_path)
    for i in range(len(test_images_empty)):
        image = cv2.resize(test_images_empty[i], (48, 48))
        img = image / 255.
        image = np.expand_dims(img, axis=0)
        class_predicted = model.predict(image)
        inID = np.argmax(class_predicted[0])
        y_true += [0]
        y_pre += [inID]
    for i in range(len(test_images_occupied)):
        image = cv2.resize(test_images_occupied[i], (48, 48))
        img = image / 255.
        image = np.expand_dims(img, axis=0)
        class_predicted = model.predict(image)
        inID = np.argmax(class_predicted[0])
        y_true += [1]
        y_pre += [inID]
    report = metrics.classification_report(y_true, y_pre, target_names=['empty','occupied'], digits=4)
    print(report)



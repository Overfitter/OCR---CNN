import numpy as np
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')


###########################################
# Load Test Set and the LabelEncoder - 01
###########################################

import cPickle
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

with open(r"lencoder.pkl", "rb") as input_file:
    le = cPickle.load(input_file)

#########################################
# Load Model - 02
#########################################

from keras.models import load_model

cnn_model = load_model('cnn_best_model.h5')


############################
# Predict the classes - 03
############################


predicted_classes = cnn_model.predict_classes(x_test,verbose=1,batch_size=32)

y_test_classes = np.argmax(y_test.astype(int),axis=-1)


########################################################################
# Make confusion matrix, plot it and also print the accuracy score - 04
########################################################################

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test_classes, predicted_classes)

plot_confusion_matrix(cm,classes=le.classes_)

print "+++++++++++++++++++++++++++++"
print "+++++++++++++++++++++++++++++"
print "ACCURACY:", accuracy_score(y_test_classes,predicted_classes)
print "+++++++++++++++++++++++++++++"
print "+++++++++++++++++++++++++++++"

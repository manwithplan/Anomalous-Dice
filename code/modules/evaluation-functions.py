import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def print_stats(predictions, labels):
    """
    Coth prints the stats of the prediction, and returns them as an array

    :predictions: list containing predictions
    :labels: list containing labels
    :return: list containing floats f1, accuracy, precision, recall
    """
    f1 = f1_score(labels, predictions)
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions)
    rec = recall_score(labels, predictions)

    print("f1 = {}".format(f1_score(labels, predictions)))
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))

    return [f1, acc, prec, rec]


def SSIMLoss(y_true, y_pred):
    """
    numpy version of the SSIM loss function

    :y_true: the original image before being processed
    :y_pred: the same image after being processed by the autoencoder moder
    :return: float between 0 and 1 giving a measure of distance between the 2 parameters
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def predictor(ano_pred, norm_pred, threshold):
    """
    predictor classifies prediction by comparing with a threshold

    :ano_pred: a list containing the predictions for anomalies between 0 and 1
    :norm_pred: a list containing the predictions for the normal samples between 0 and 1
    :return; 2 lists, 1 containing prediction and the other containing the labels.
    """
    threshold = np.float32(threshold)
    predictions = []
    labels = []

    for pred in ano_pred:
        labels.append(1)
        if pred > threshold:
            predictions.append(1)
        else:
            predictions.append(0)

    for pred in norm_pred:
        labels.append(0)
        if pred > threshold:
            predictions.append(1)

        else:
            predictions.append(0)

    return predictions, labels


def make_predictions(model, anomalies, normals):
    """
    Function implementing SSIMloss on the both anomalies and normal samples

    :model: autoencoder or any other model taking in 128x128 images
    :anomalies: list of np arrays of 128 x 128
    :normals: list of np arrays of 128 x 128
    :return: a list containing 4 metrics, f1, accuracy, precision, recall
    """
    ano_predictions = []
    norm_predictions = []

    result = model.predict(anomalies).reshape(128, 128, 1)
    ano_predictions.append(SSIMLoss(anomalies, result).numpy())

    result = model.predict(normals).reshape(128, 128, 1)
    norm_predictions.append(SSIMLoss(normals, result).numpy())

    threshold = (
        np.mean(norm_predictions)
        + (np.mean(norm_predictions) + np.mean(ano_predictions)) / 12
    )

    predictions, labels = predictor(ano_predictions, norm_predictions, threshold)

    return print_stats(predictions, labels)

## Autoencoder with cropped and rotated data.

- 100 samples from each class

model with 3 convolution layer and 3 deconvolution layers 
loss function - SSIMLoss
epoc count - 10

learning rate - 0.0005

    f1 = 0.6722689075630252
    Accuracy = 0.7784090909090909
    Precision = 0.6349206349206349
    Recall = 0.7142857142857143

learning rate - 0.0001

    f1 = 0.2469135802469136
    Accuracy = 0.8665207877461707
    Precision = 0.16042780748663102
    Recall = 0.5357142857142857

- All samples from each class

model with 3 convolution layer and 3 deconvolution layers 
loss function - SSIMLoss
epoc count - 10

calculated threshold with mean from all classes

    f1 = 0.15358931552587646
    Accuracy = 0.6301969365426696
    Precision = 0.0847145488029466
    Recall = 0.8214285714285714

calculated threshold based on different classes

learning rate - 0.0005

    f1 = 0.46808510638297873
    Accuracy = 0.9635302698760029
    Precision = 0.5789473684210527
    Recall = 0.39285714285714285

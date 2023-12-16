'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np
import keras

output_dim = 10      # output layer dimensionality = num_classes
'''
D2. Load MNIST data / Only for Toy Project
'''

# print(tf.__version__)
## MNIST Dataset #########################################################
# mnist = tf.keras.datasets.mnist
# class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
##########################################################################

## Fashion MNIST Dataset #################################################
mnist = tf.keras.datasets.fashion_mnist
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
##########################################################################
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Change data type as float. If it is int type, it might cause error 
'''
D3. Data Preprocessing
'''
# Normalizing
X_train, X_test = X_train / 255.0, X_test / 255.0

print(Y_train[0:10])
print(X_train.shape)

'''
D4. EDA(? / Exploratory data analysis)
'''

# plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    # if you want to invert color, you can use 'gray_r'. this can be used only for MNIST, Fashion MNIST not cifar10
    # pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray_r'))
    
# show the figure
plt.show()

# convert class vectors to binary class matrices
Y_train = keras.utils.to_categorical(Y_train, output_dim)
Y_test  = keras.utils.to_categorical(Y_test, output_dim)

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, GlobalAveragePooling2D, Flatten
from keras import optimizers

# Here we make our model by randomly assigning values to various parameters

def create_model(filter_size_1, filter_size_2, filter_size_3,
                 # kernel_size_1, kernel_size_2,
                 act_func_1, act_func_2):

    """Builds a Sequential CNN model to recognize MNIST.

    Args:
      input_shape: Shape of the input depending on the `image_data_format`.
      dropout2_rate: float between 0 and 1. Fraction of the input units to drop for `dropout_2` layer.
      dense_1_neurons: Number of neurons for `dense1` layer.

    Returns:
      a Keras model

    """
    # Reset the tensorflow backend session.
    # tf.keras.backend.clear_session()
    # Define a CNN model to recognize MNIST.
    model = Sequential()
    
    # Normal CNN Layer with Stride=1, Padding=SAME, 1<=kernel<8, and any of one activation function
    model.add(Conv2D(filters=filter_size_1, kernel_size=(2,2),
                     activation=act_func_1, strides=(1,1), padding='SAME', input_shape=[28,28,1]))
    
    # Reduction CNN Layer 1 with Stride=2, Padding=VALID, 1<=kernel<8, and any of one activation function
    model.add(Conv2D(filters=filter_size_2, kernel_size=(2,2),
                     activation=act_func_2, strides=(2,2), padding='valid'))
    
    # Reduction CNN Layer 2 with Stride=2, Padding=VALID, 1<=kernel<8, and any of one activation function
    model.add(Conv2D(filters=filter_size_3, kernel_size=(2,2),
                     activation=act_func_2, strides=(2,2), padding='valid'))
    
    # Final CNN layer with Globel Average Pooling
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.25, name="dropout_1"))
    
    # Flatten our data
    model.add(Flatten())
    
    # In order to avoid overfitting, using dropout
    # model.add(Dropout(rate = dropout_1))
    
    # Fully connected dense layer with 64 units & an activation function
    model.add(Dense(units = 64, activation = 'relu'))
    
    # Output with 10 layers ranging from 0 to 9, each indicating the type of cloth
    model.add(Dense(output_dim, activation= 'softmax'))

    # Returning our model 
    return model


learning_rate = 5e-4

# Step 2.1: Finding out the fitness values of each of the individual people in our population 
def fit_params(model, epochs, optimizer):
    
    # Compiling our cnn model which we make above with layers
    # optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    # Fitting our model
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size = 2048, verbose=2)

    # Evaluate the model with the eval dataset.
    score = model.evaluate(X_test, Y_test)
    print('Test loss    :', score[0])
    print('Test accuracy:', score[1])

    #returning 1st index value of res, because it contains value of accuracy, and 0th index contains value of loss
    return score[1]

from random import choice
from random import uniform
from numpy.random import randint


number_of_trials = 10


# Making a dictionary which contains key->value pairs which contains key as the accuracy & value as the paramters of our cnn model which gets that accuracy
track_of_param_and_accuracy={}

# This indicates the maximum accuracy value along with its parameters
highest_acc_with_param=0

# This contains the list of all the accuracies we got
list_of_accuracies = []

for idx in range(number_of_trials):
  
    # This indicates the number of paramater in our cnn model
    cnn_model_parameters = {}

    # This will choose normal CNN Layer Filters
    filter_size_1 = choice([32, 128, 8, 256,16, 64])
    cnn_model_parameters["filter_size_1"] = filter_size_1

    # This will choose Reduction CNN Layer 1 Filters
    get = choice([256, 128, 8, 32,16, 64])
    filter_size_2=get
    cnn_model_parameters["filter_size_2"] = filter_size_2

    # This will choose Reduction CNN Layer 2 Filters
    filter_size_3 = get
    cnn_model_parameters["filter_size_3"] = filter_size_3

    # This will choose Normal CNN Layer Kernal matrix
    # kernel_size_1 = choice([1,2,3,4,5,6,7])
    # cnn_model_parameters["kernel_size_1"] = kernel_size_1

    # This will choose Reduction CNN Layer Kernal matrix
    # kernel_size_2 = choice([1,2,3,4,5,6,7])
    # cnn_model_parameters["kernel_size_2"] = kernel_size_2

    # This will choose Normal CNN Layer Activation Function
    act_func_1 = choice(["relu", "sigmoid", "tanh", "swish", "gelu"])
    cnn_model_parameters["act_func_1"] = act_func_1

    # This will choose Reduction CNN Layer Activation Function
    # For both of our Reduction CNN layers, the activation function, number of filters, kernel matrix size, all are same
    act_func_2 = choice(["relu", "sigmoid", "tanh", "swish", "gelu"])
    cnn_model_parameters["act_func_2"] = act_func_2

    # Optimizer is same
    optmzr = choice(['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'])
    cnn_model_parameters["optmzr"] = optmzr

    #epochs can be any integer between 10 and 30
    epochs = randint(10, 40)
    cnn_model_parameters["epochs"] = epochs

    # With those values making our model
    cnn_model = create_model(filter_size_1, filter_size_2, filter_size_3,
                             # kernel_size_1, kernel_size_2,
                             act_func_1, act_func_2)
    # Finding out the fitness of our model
    gotten_accuracy = fit_params(cnn_model, epochs, optmzr)
    
    print("Trial ", idx, "Results")
    print("The parameters of CNN Model are: ", cnn_model_parameters)
    print("The accuracy of CNN Model is: ", round(gotten_accuracy,3), "\n")

    # This will contain like key->value pair, which tells the accuracy and its corresponding parameters
    track_of_param_and_accuracy[gotten_accuracy]= cnn_model_parameters

    # This will pull out that one who has maximum accuracy
    highest_acc_with_param=max(track_of_param_and_accuracy)

    # Appending that accuracy in our list
    list_of_accuracies.append(gotten_accuracy)

arr=[]
for value in track_of_param_and_accuracy.get(highest_acc_with_param).values():
    arr.append(value)

print("----------- Optimized Architecture \n",
      "Convolution 1 Layer filter size : ", arr[0], '\n', 
      "Activation function 1           : ", arr[3], '\n', 
      "Convolution 2 Layer filter size : ", arr[1], '\n', 
      "Activation function 2           : ", arr[4], '\n', 
      "Convolution 3 Layer filter size : ", arr[2], '\n', 
      "Activation function 3           : ", arr[4], '\n', 
      "optimizer                       : ", arr[5], '\n', 
      "Number of epochs                : ", arr[6] 
     )


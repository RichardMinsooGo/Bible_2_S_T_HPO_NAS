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

def create_model( act_func ):

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
    model.add(Conv2D(filters=128, kernel_size=(2,2),
                     activation=act_func, strides=(1,1), padding='SAME', input_shape=[28,28,1]))
    
    # Reduction CNN Layer 1 with Stride=2, Padding=VALID, 1<=kernel<8, and any of one activation function
    model.add(Conv2D(filters=128, kernel_size=(2,2),
                     activation=act_func, strides=(2,2), padding='valid'))
    
    # Reduction CNN Layer 2 with Stride=2, Padding=VALID, 1<=kernel<8, and any of one activation function
    model.add(Conv2D(filters=128, kernel_size=(2,2),
                     activation=act_func, strides=(2,2), padding='valid'))
    
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
def fit_params(model, optimizer):
    
    # Compiling our cnn model which we make above with layers
    # optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    # Fitting our model
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size = 2048, verbose=2)

    # Evaluate the model with the eval dataset.
    score = model.evaluate(X_test, Y_test)
    print('Test loss    :', score[0])
    print('Test accuracy:', score[1])

    #returning 1st index value of res, because it contains value of accuracy, and 0th index contains value of loss
    return score[1]



# Making a dictionary which contains key->value pairs which contains key as the accuracy & value as the paramters of our cnn model which gets that accuracy
track_of_param_and_accuracy={}

# This indicates the maximum accuracy value along with its parameters
highest_acc_with_param=0

# This contains the list of all the accuracies we got
list_of_accuracies = []

# This will choose Reduction CNN Layer Activation Function
# For both of our Reduction CNN layers, the activation function, number of filters, kernel matrix size, all are same
act_func = ["relu", "sigmoid", "tanh", "swish", "gelu"]
# Optimizer is same
optmzr   = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

idx = 0
for idx_a in range(len(act_func)):
    for idx_b in range(len(optmzr)):
        idx += 1
  
        # This indicates the number of paramater in our cnn model
        cnn_model_parameters = {}

        # This will choose Normal CNN Layer Activation Function
        cnn_model_parameters["act_func "] = act_func[idx_a]

        # Optimizer is same
        cnn_model_parameters["optmzr"] = optmzr[idx_b]


        # With those values making our model
        cnn_model = create_model(act_func[idx_a])
        
        # Finding out the fitness of our model
        gotten_accuracy = fit_params(cnn_model,optmzr[idx_b])

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
      "Activation function : ", arr[0], '\n', 
      "optimizer           : ", arr[1]
     )


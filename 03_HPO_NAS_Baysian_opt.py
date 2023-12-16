! pip install bayesian-optimization

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

def create_model(filter_size_1, filter_size_2, filter_size_3):

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
                     activation='relu', strides=(1,1), padding='SAME', input_shape=[28,28,1]))
    
    # Reduction CNN Layer 1 with Stride=2, Padding=VALID, 1<=kernel<8, and any of one activation function
    model.add(Conv2D(filters=filter_size_2, kernel_size=(2,2),
                     activation='relu', strides=(2,2), padding='valid'))
    
    # Reduction CNN Layer 2 with Stride=2, Padding=VALID, 1<=kernel<8, and any of one activation function
    model.add(Conv2D(filters=filter_size_3, kernel_size=(2,2),
                     activation='relu', strides=(2,2), padding='valid'))
    
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
def fit_params(verbose, filter_size_1, filter_size_2, filter_size_3, epochs):

    # Create the model using a specified hyperparameters.
    filter_size_1 = int(filter_size_1)
    filter_size_2 = int(filter_size_2)
    filter_size_3 = int(filter_size_3)
    epochs = int(epochs)
    model = create_model(filter_size_1, filter_size_2, filter_size_3)
    
    # Compiling our cnn model which we make above with layers
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    # Fitting our model
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size = 2048, verbose=2)

    # Evaluate the model with the eval dataset.
    score = model.evaluate(X_test, Y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    #returning 1st index value of res, because it contains value of accuracy, and 0th index contains value of loss
    return score[1]


from functools import partial

verbose = 1
fit_params_partial = partial(fit_params, verbose)

fit_params_partial(filter_size_1 = 64, filter_size_2 = 64, filter_size_3 = 64, epochs = 25)


# The BayesianOptimization object will work out of the box without much tuning needed. The main method you should be aware of is `maximize`, which does exactly what you think it does.
#
# There are many parameters you can pass to maximize, nonetheless, the most important ones are:
# - `n_iter`: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
# - `init_points`: How many steps of **random** exploration you want to perform. Random exploration can help by diversifying the exploration space.

# In[11]:


from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {'filter_size_1': (8, 256), 'filter_size_2': (8, 256), 'filter_size_3': (8, 256), "epochs": (10, 40)}

optimizer = BayesianOptimization(
    f=fit_params_partial,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

# Start the parameter search
optimizer.maximize(init_points=3, n_iter=9)

# Print the results for each iteration
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

# Print the result of the best model
print(optimizer.max)




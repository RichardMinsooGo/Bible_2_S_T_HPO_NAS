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
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    #returning 1st index value of res, because it contains value of accuracy, and 0th index contains value of loss
    return score[1]

from random import choice
from random import uniform
from numpy.random import randint

# This code will take random values of the parameter which will get assign to our CNN model

def assign_random_values_to_parameters():
  
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

    return cnn_model_parameters

# Genetic Algorithm Step 1, making initial population so that we can able to start the reproduction phase

#Passed initial number of peoples in population
def making_initial_population(number_of_people_in_population):
    peoples_with_parameters = []
    for i in range(number_of_people_in_population):
        # For each person assigning it with random values of parameters
        cnn_model_parameters = assign_random_values_to_parameters()
        #Appending those values to peoples
        peoples_with_parameters.append(cnn_model_parameters)
    return peoples_with_parameters

# Step 2.2: Selecting the best individual out of all the individuals from our population by using Roulette wheel selection method
def rltwhlslct(fitness_of_population):
    # Finding out the total fitness score of our whole population
    total = sum(fitness_of_population)

    # Finding out the weightage of fitness score of each of individual in our population
    weightage = [round((x/total) * 100) for x in fitness_of_population]

    #This is the wheel which we rotate, and then on the basis where the pointer points in the wheel, that individual will be choosen
    roulette_pointer_wheel = []

    #Here enumerting the weightage array, so that we get values in the form of key->value pairs with key starting from 0
    for sample_space_ind,value in enumerate(weightage):

        #Here we are appending the output in our roulette_pointer_wheel array
        roulette_pointer_wheel.extend([sample_space_ind]*value)

    #Sorting the weights
    roulette_pointer_wheel.sort()
    # Taking the maximum as our 1st parent
    parent1 = roulette_pointer_wheel[-1]
    # Taking the second maximum as our 2nd parent
    parent2 = roulette_pointer_wheel[-2]
    # Returning those 2 parents
    return [parent1, parent2]

#Step 3: Doing Cross-over of the 2 fittest individual from our model

# Here we passed both the parents along with there parameter values
def parent_cross_over(first_parent, second_parent):
  
    # From this function we expect 2 childrens to return
    first_child = {}
    # Making dictionary of both the children
    second_child = {}

    # First child's filter_size_1 can be any of 1st parent's or 2nd parent's filter_size_1
    first_child["filter_size_1"] = choice([first_parent["filter_size_1"], second_parent["filter_size_1"]])

    # First child's filter_size_2 can be any of 1st parent's or 2nd parent's filter_size_2
    first_child["filter_size_2"] = choice([first_parent["filter_size_2"], second_parent["filter_size_2"]])

    # First child's filter_size_3 can be any of 1st parent's or 2nd parent's filter_size_3
    first_child["filter_size_3"] = choice([first_parent["filter_size_3"], second_parent["filter_size_3"]])

    # Second child's filter_size_2 can be any of 1st parent's or 2nd parent's filter_size_2
    second_child["filter_size_1"] = choice([first_parent["filter_size_1"], second_parent["filter_size_1"]])

    # Second child's filter_size_3 can be any of 1st parent's or 2nd parent's filter_size_3
    second_child["filter_size_2"] = choice([first_parent["filter_size_2"], second_parent["filter_size_2"]])
    second_child["filter_size_3"] = choice([first_parent["filter_size_3"], second_parent["filter_size_3"]])

    # First child's normalkernellayer can be any of 1st parent's or 2nd parent's normalkernellayer
    # first_child["kernel_size_1"]  = choice([first_parent["kernel_size_1"], second_parent["kernel_size_1"]])

    # Second child's normalkernellayer can be any of 1st parent's or 2nd parent's normalkernellayer
    # second_child["kernel_size_1"] = choice([first_parent["kernel_size_1"], second_parent["kernel_size_1"]])

    # First child's reductionkernellayer can be any of 1st parent's or 2nd parent's reductionkernellayer
    # first_child["kernel_size_2"] = choice([first_parent["kernel_size_2"], second_parent["kernel_size_2"]])

    # Second child's reductionkernellayer can be any of 1st parent's or 2nd parent's reductionkernellayer
    # second_child["kernel_size_2"] = choice([first_parent["kernel_size_2"], second_parent["kernel_size_2"]])

    # Here Doing the Crossover of parent features in child, so as to get best results
    # First child's normal activation function is 1st parent's reduction activation function
    first_child["act_func_1"] = first_parent["act_func_2"]

    # Second child's normal activation function is 2nd parent's reduction activation function
    second_child["act_func_1"] = second_parent["act_func_2"]

    # First child's reduction activation function is 1st parent's normal activation function
    first_child["act_func_2"] = second_parent["act_func_1"]

    # Second child's reduction activation function is 1st parent's normal activation function
    second_child["act_func_2"] = first_parent["act_func_1"]

    # Again cross-oover here so to get diverse characteristics
    # First child's optimizer is 2nd parent's optimizer
    first_child["optmzr"] = second_parent["optmzr"]

    # Second child's optimizer is 1st parent's optimizer
    second_child["optmzr"] = first_parent["optmzr"]

    # First child's epochs is 1st parent's epochs
    first_child["epochs"] = first_parent["epochs"]

    # Second child's epochs is 2nd parent's epochs
    second_child["epochs"] = second_parent["epochs"]

    # Returing both the offsprings which we get here by doing cross-over of the 2 parents above
    return [first_child, second_child]

# Step 4: In this we mutating the child so as to get the diverse characteristics in the child
def child_mutn(individual):
    answr = randint(0,50)
    if answr <= 15:
        # Randomly inreasing the number of epochs so as to make it more
        individual["epochs"] += randint(0, 20)
    return individual

# Making our model to run for the given number of generations & giving the size of initial population

# Indicates the total number of generations
number_of_generations = 2

# Indicates the number of peoples in our population
number_of_people_in_population = 4

# Making a dictionary which contains key->value pairs which contains key as the accuracy & value as the paramters of our cnn model which gets that accuracy
track_of_param_and_accuracy={}

# This indicates the maximum accuracy value along with its parameters
highest_acc_with_param=0

# Making our population with the given number of individuals and here we get all peoples with parameters
peoples = making_initial_population(number_of_people_in_population)

# Here we are iterating each of generation so as to get result in each generation
for each_gen in range(number_of_generations):
    print("Currently Runnung Generation is:",each_gen+1)

    # This contains the list of all the accuracies we got
    list_of_accuracies = []
    # Iterating for each of our individual person in the population
    for individual in peoples:
        # Taking values of each parameter from the peoples array

        # Taking value of normal filter
        filter_size_1 = individual["filter_size_1"]

        # Taking value of reduction layer 1 filter
        filter_size_2 = individual["filter_size_2"]

        # Taking value of reduction layer 2 filter
        filter_size_3 = individual["filter_size_3"]

        # Taking value of normal layer kernel
        # kernel_size_1 = individual["kernel_size_1"]

        # Taking value of reduction layer kernel
        # kernel_size_2 = individual["kernel_size_2"]

        # Taking value of normal activation function
        act_func_1    = individual["act_func_1"]

        # Taking value of redcution activation function
        act_func_2    = individual["act_func_2"]
        # dropout_1    = individual["dropout_1"]
        # dropout_2    = individual["dropout_2"]

        # Taking value of optimizer
        optmzr        = individual["optmzr"]

        # Taking value of epochs
        epochs        = individual["epochs"]

        # With those values making our model
        cnn_model = create_model(filter_size_1, filter_size_2, filter_size_3,
                                 # kernel_size_1, kernel_size_2,
                                 act_func_1, act_func_2)
        # Finding out the fitness of our model
        gotten_accuracy = fit_params(cnn_model, epochs, optmzr)
        print("The parameters of CNN Model are: ", individual)
        print("The accuracy of CNN Model is: ", round(gotten_accuracy,3))

        # This will contain like key->value pair, which tells the accuracy and its corresponding parameters
        track_of_param_and_accuracy[gotten_accuracy]= individual

        # This will pull out that one who has maximum accuracy
        highest_acc_with_param=max(track_of_param_and_accuracy)

        # Appending that accuracy in our list
        list_of_accuracies.append(gotten_accuracy)

    # Sending the list of all accuracies in roulette_pointer_wheel function & returning the fittest 2 parents
    two_fittest_persons = rltwhlslct(list_of_accuracies)
    parent1 = peoples[two_fittest_persons[0]]
    parent2 = peoples[two_fittest_persons[1]]

    # Sending those 2 parents with parameters to cross-over function so as to produce an offspring
    childs = parent_cross_over(parent1, parent2)

    # Doing the mutation of each of the child, so as to get the diverse characteristics in the child
    child1 = child_mutn(childs[0])
    child2 = child_mutn(childs[1])

    # Now appending those childs in our population, as they will lead the new generation
    peoples.append(child1)
    peoples.append(child2)

    # Removing those who have least 2 accuracies
    # Getting the 1st least one
    least1 = min(list_of_accuracies)
    # Getting index of 1st least
    least1_index = list_of_accuracies.index(least1)
    # Removing the 1st least
    peoples.remove(peoples[least1_index])
    # Getting the 2nd least one
    least2 = min(list_of_accuracies)
    # Getting index of 2nd least
    least2_index = list_of_accuracies.index(least2)
    # Removing the 2nd least
    peoples.remove(peoples[least2_index])

# print(track_of_param_and_accuracy.get(max_key))

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


'''
This programme predicts iris species from measurements given by users.
'''

import pickle
import numpy as np

# load model from file
filename = 'model.pickle'
model = pickle.load(open(filename, 'rb'))

# dictionary for translation
species_dict = {0:'setosa', 1:'versicolor', 2:'virginica'}

# use the model for a single prediction
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    # reformat input to satisfy sci-kit learn's preferences
    measurements = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
    # use model
    prediction = int(model.predict(measurements))
    # translate model output to species using dictionary
    species = species_dict[prediction]
    return species

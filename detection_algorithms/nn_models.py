
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras import metrics

from sklearn.utils import class_weight
from scipy.stats import pearsonr

def create_neural_network_2layer_model(input_size, output_size):
# def create_neural_network_5layer_model():

    print('input_size: '.format(input_size))
    print('output_size: '.format(output_size))

    model = Sequential()

    # model.add(Dense(input_size, input_dim=X.shape[1], init='uniform', activation='relu'))
    model.add(Dense(input_size, input_dim=input_size, init='uniform', activation='relu'))

    # model.add(Dense(y_new.shape[1], init='uniform', activation='relu'))
    model.add(Dense(output_size, init='uniform', activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
    return model


def create_neural_network_3layer_model(input_size, output_size):
# def create_neural_network_5layer_model():

    print('input_size: '.format(input_size))
    print('output_size: '.format(output_size))

    model = Sequential()

    # model.add(Dense(input_size, input_dim=X.shape[1], init='uniform', activation='relu'))
    model.add(Dense(input_size, input_dim=input_size, init='uniform', activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(400, init='uniform', activation='relu'))

    # model.add(Dense(y_new.shape[1], init='uniform', activation='relu'))
    model.add(Dense(output_size, init='uniform', activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
    return model



def create_neural_network_5layer_model(input_size, output_size):
# def create_neural_network_5layer_model():

    print('input_size: '.format(input_size))
    print('output_size: '.format(output_size))

    model = Sequential()

    # model.add(Dense(input_size, input_dim=X.shape[1], init='uniform', activation='relu'))
    model.add(Dense(input_size, input_dim=input_size, init='uniform', activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(400, init='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200, init='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, init='uniform', activation='relu'))

    # model.add(Dense(y_new.shape[1], init='uniform', activation='relu'))
    model.add(Dense(output_size, init='uniform', activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
    return model


# def create_neural_network_4layer_model():
def create_neural_network_4layer_model(input_size, output_size):

    print('input_size: '.format(input_size))
    print('output_size: '.format(output_size))
    model = Sequential()

    # model.add(Dense(input_size, input_dim=X.shape[1], init='uniform', activation='relu'))
    model.add(Dense(input_size, input_dim=input_size, init='uniform', activation='relu'))


    model.add(Dropout(0.2))
    model.add(Dense(200, init='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, init='uniform', activation='relu'))

    # model.add(Dense(y_new.shape[1], init='uniform', activation='relu'))
    model.add(Dense(output_size, init='uniform', activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
    return model


# def create_neural_network_4layer_model():
def create_neural_network_4layer_no_dropout_model(input_size, output_size):

    print('input_size: '.format(input_size))
    print('output_size: '.format(output_size))
    model = Sequential()

    # model.add(Dense(input_size, input_dim=X.shape[1], init='uniform', activation='relu'))
    model.add(Dense(input_size, input_dim=input_size, init='uniform', activation='relu'))

    model.add(Dense(200, init='uniform', activation='relu'))
    model.add(Dense(100, init='uniform', activation='relu'))

    # model.add(Dense(y_new.shape[1], init='uniform', activation='relu'))
    model.add(Dense(output_size, init='uniform', activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
    return model
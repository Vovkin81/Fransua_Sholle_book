from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

model = build_model()
history = model.fit(train_data, train_labels,
                    epochs=500,
                    batch_size=1,
                    validation_data=(test_data, test_labels),
                    verbose=0)

# graph
history_dict = history.history
loss_values = history_dict['mean_absolute_error']
val_loss_values = history_dict['val_mean_absolute_error']

epochs = range(1, len(history_dict['mean_absolute_error'])+1)
plt.plot(epochs, loss_values, 'black', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Trainig and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
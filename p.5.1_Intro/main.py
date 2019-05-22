from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

print("train_data.shape:", train_data.shape)
print("test_data.shape:", test_data.shape)

train_data = train_data.reshape((60000, 28, 28, 1))
train_data = train_data.astype('float32') / 255
test_data = test_data.reshape((10000, 28, 28, 1))
test_data = test_data.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print("train_data.shape:", train_data.shape)
print("test_data.shape:", test_data.shape)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

print("model.summary: ", model.summary())

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels, epochs=20, batch_size=64,
                    validation_data=(test_data, test_labels), verbose=2)

# history = model.fit(x_train, y_train, epochs=10, batch_size=512, validation_data=(x_test, y_test), verbose=2)
# history = model.fit(x_train, one_hot_train_labels, epochs=20, batch_size=512, validation_data=(x_test, one_hot_test_labels), verbose=2)

# graph
history_dict = history.history
loss_values = history_dict['acc']
val_loss_values = history_dict['val_acc']

epochs = range(1, len(history_dict['acc'])+1)
plt.plot(epochs, loss_values, 'black', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Trainig and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
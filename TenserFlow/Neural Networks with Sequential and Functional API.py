import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Load and preprocess the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# Sequential API (very convenient, not very flexible)
model = keras.Sequential(
    [
        keras.Input(shape =(28*28,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10),
    ]
)
# sequential
model = keras.Sequential()
model.add(keras.Input(shape=(784,))) # adding  input layer
model.add(layers.Dense(256,activation='relu'))
# model.add(layers.Dense(256,activation= 'relu'))
model.add(layers.Dense(256,activation= 'relu',name='my_layer'))
model.add(layers.Dense(10))

# model  = keras.Model(inputs = model.inputs,
#                      outputs=[model.layers[-2].output]) 3 1st method to call layer

# model  = keras.Model(inputs = model.inputs,
#                      outputs=[model.get_layer('my_layer').output]) # 2nd method to call layer


model  = keras.Model(inputs = model.inputs,
                     outputs=[layer.output for layer in model.layers]) # method to call all  layer

features = model.predict(x_train)
for feature in features:
    print(feature.shape)

import  sys
sys.exit()


# functional API (A bit more flexible)
inputs = keras.Input(shape=(784,))
x = layers.Dense(512,activation='relu',name='first_layer')(inputs)
x = layers.Dense(256,activation='relu',name='second_layer')(x)
outputs = layers.Dense(10,activation='softmax')(x)
model = keras.Model(inputs=inputs,outputs=outputs)

print(model.summary())
# Compile the model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)

# Evaluate the model
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
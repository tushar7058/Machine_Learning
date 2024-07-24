import inputs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Define the model
model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, padding='valid', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10),
    ]
)

# Print the model summary
model.summary()


def my_model():
    input = keras.Input(shape=(32,32,2)),
    x = layers.Conv2D(32,3,)(input)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Maxpooling2D()(x)
    x = layers.Conv2D(64,5,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x =keras.activations.relu(x)
    x = keras.Conv2D(128,3)(x)
    x =layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = keras.Flatten()
    x = layers.Dense(64,activation='relu')(x)

    outputs = layers.Dense(10)(x)
    model = keras.model(inputs=inputs,outputs=outputs),
    return  model




# Compile the model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"],
)

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)

# Evaluate the model
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
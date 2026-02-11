import tensorflow as tf
print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() #fetches data from google cloud storage bucket

x_train, x_test = x_train / 255.0, x_test / 255.0

"""
Flatten: flattens 28x28 matrix into array of 784 numbers
Dense: connects all 784 inputs to 128 neurons
dropout: randomly shuts off 20% of neurons to avoid overfitting
Dense: connect 128 neurons into 10 final neurons
"""
model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

predictions = model(x_train[:1]).numpy()
print(predictions)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
    ])

probability_model(x_test[:5])

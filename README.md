# tensorflow
Explore Tensorflow

1. Image classification using fastion dataset
    1. Import required libs
    2. Dataset download 'tf.keras.datasets.fashion_mnist'
    3. classnames i.e. Outputs
    4. Preprocess data
    5. Building Model 'tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)), tf.keras.layers.Dense(128, activation='relu'),tf.keras.layers.Dense(10)])'
    6. Compile the Model 'model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])'
    7. Training the Model 'model.fit(train_images,train_lables,epochs=10)'
      1.Input data into the model
      2.Making connections
      3.Making predictions
      4.Verification
    8. Accuracy Evaluation 'test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)'
    9. Making Prediction 'probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])'
    10. Verify Prediction 'predictions = probability_model.predict(test_images)'   'np.argmax(predictions[10]) == test_labels[10]'

import tensorflow as tf
import pandas as pd

train_data = pd.read_csv('data/wdbc-train.csv')
test_data = pd.read_csv('data/wdbc-test.csv')

x_train = train_data.drop(columns=['diagnosis'])
y_train = train_data['diagnosis']

x_test = test_data.drop(columns=['diagnosis'])
y_test = test_data['diagnosis']

numOfFeatures = len(x_train.columns)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation='relu', input_shape=(numOfFeatures,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(0.06), metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), verbose=1)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')


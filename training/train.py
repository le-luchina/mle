import os
import sys

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from data_process.data_split import X_train, X_test, y_train, y_test

# Add validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Build the model
model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu', name='fc1'))
model.add(Dense(10, activation='relu', name='fc2'))
model.add(Dense(3, activation='softmax', name='output'))

# Compile model with Adam optimizer
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping with patience of 5 epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with validation data and early stopping
history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    verbose=2, batch_size=5, epochs=200, callbacks=[early_stopping])

# Train the model with validation data and early stopping
history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    verbose=2, batch_size=5, epochs=200, callbacks=[early_stopping])

# Evaluation on training set
results = model.evaluate(X_train, y_train)

print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))

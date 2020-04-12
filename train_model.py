import numpy as np
from model import crnn
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# Load spectrograms
eminem_spectrogram_folder = ''
not_eminem_spectrogram_folder = ''

eminem_specs = np.load(eminem_spectrogram_folder)
not_eminem_specs = np.load(not_eminem_spectrogram_folder)

# Make train and test set
X = np.concatenate((eminem_specs, not_eminem_specs), axis=0)
y = np.array([1] * len(eminem_specs) + [0] * len(not_eminem_specs))
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)

# Create crnn model
lr = 0.0001
input_shape = (X.shape[1], X.shape[2], X.shape[3])
model = crnn(input_shape=input_shape)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

# Train model
batch_size = 16
epochs = 10
print(model.summary())
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate on test set
model.evaluate(X_test, y_test)

#Save model
model_save_name = 'model.h5'
model.save(model_save_name)

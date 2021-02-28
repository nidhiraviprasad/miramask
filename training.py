from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model

trainingPath = 'training'
testPath = 'test/dataset'
trainingSet = ImageDataGenerator(rescale=1./225).flow_from_directory(trainingPath, color_mode='grayscale', target_size=(150, 150), shuffle=True, classes=['with_mask', 'without_mask'])
testSet = ImageDataGenerator(rescale=1./225).flow_from_directory(testPath, color_mode='grayscale', target_size=(150, 150), shuffle=True, classes=['with_mask', 'without_mask'])

width, height = 150, 150
model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape=(width, height,1)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(width, height,1)))
model.add(MaxPooling2D(2,2))


model.add(Conv2D(128, (3,3), activation='relu', input_shape=(width, height,1)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128, (3,3), activation='relu', input_shape=(width, height,1)))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(120, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit_generator(trainingSet, epochs=10, validation_data=testSet, verbose=1, shuffle=True)

model.save('mask.h5')

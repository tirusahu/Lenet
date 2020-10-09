
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import keras

# Initialising the CNN
#classifier = Sequential()
model = Sequential()

# Step 1 - Convolution
#classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(64, 64, 3)))

# Step 2 - Pooling
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# Step 3 - Flattening
#classifier.add(Flatten())
model.add(Flatten())

# Step 4 - Full connection
#classifier.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))

#classifier.add(Dense(units = 3, activation = 'softmax'))
model.add(Dense(3, activation='softmax'))

# Compiling the CNN
#classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/Users/user/Desktop/ImageScrapper/images1',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/Users/user/Desktop/ImageScrapper/images1',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model1 = model.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = 200)


model.save("/Users/user/Desktop/ImageScrapper/images1/Lenetmodel.h5")
print("Saved model to disk")

# Part 3 - Making new predictions




import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/Users/user/Desktop/ImageScrapper/images/parrots/jpg_30.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Parrot'
    print(prediction)
else:
    prediction = 'Peacock'
    print(prediction)
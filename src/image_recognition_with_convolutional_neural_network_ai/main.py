import os
from data_processing import get_data_generators
from cnn import build_model
from tensorflow.keras.callbacks import ModelCheckpoint

DATASET_PATH = 'dataset'
MODEL_PATH = 'model/cnn_model.h5'
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 100

train_gen, val_gen = get_data_generators(DATASET_PATH, IMAGE_SIZE, BATCH_SIZE)

model = build_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=train_gen.num_classes)

checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[checkpoint])
# Just for Building Model not for using model.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
# import joblib
# Building Model
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

model.compile()
# joblib.dump(model,'../../saved_models/resnet50_model_for_embeddings.joblib');
# model.save("resnet50_model_for_embeddings.h5")


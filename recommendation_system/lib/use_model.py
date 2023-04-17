from django.templatetags.static import static
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from numpy.linalg import norm
from PIL import Image
import requests
from io import BytesIO
# import joblib
from .build_model import model

# model = joblib.load(static("resnet50_model_for_embeddings.joblib"));

def get_embiddings(url):
  try:
    # url = "https://gateway.pinata.cloud/ipfs/QmVifjUGN8g1bZxqEdxHPVZUSeReXTRSRDbzd5m4fvKrEP"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224,224))
    img_batch = np.array([image.img_to_array(img)])
    preprocessed_img_batch = preprocess_input(img_batch)
    result = model.predict(preprocessed_img_batch).flatten()
    normalized_result = result / norm(result)
    return normalized_result.tolist()
  except:
    return []


# print(type(get_embiddings()))
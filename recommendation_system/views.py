from django.shortcuts import render, HttpResponse
import json
from .lib.use_model import get_embiddings

def get_image_embeddings(request):
    json_data = json.loads(request.body.decode('utf-8'))
    if 'image_URL' in json_data:
        return HttpResponse(json.dumps({"embeddings": get_embiddings(json_data['image_URL'])}))
    return HttpResponse("<h1>Hello Recommendation system - Default</h1>")

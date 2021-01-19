"""
Definition of views.
"""
import numpy as np 
import pandas as pd 
import base64
import io
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from PIL import Image
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session
from django.shortcuts import render
global graph
global model
global session
import h5py
import requests
from tensorflow.python.keras import backend as K

import warnings
warnings.filterwarnings("ignore")


from datetime import datetime
from django.shortcuts import render
from django.http import HttpRequest
import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# launch graph in session
session = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
with session.graph.as_default():
    K.set_session(session)
    model = tf.keras.models.load_model(settings.BASE_DIR+'/app/models/tumor_prediction.h5', compile=False)

# initialize global variables
# init = tf.global_variables_initializer()
# session.run(init)

def home(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)    

    #logo_url=os.path.join(settings.BASE_DIR,'app\static\src\logo.png')
    #print(logo_url)

    return render(
        request,
        'app/index.html',
        {
            'title':'Home Page',
            'year':datetime.now().year,
        }
    )

def analysis(request, *args, **kwargs):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)

    #load image
    try:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(settings.UPLOAD_DIR+'/'+myfile.name, myfile)
        img = tf.keras.preprocessing.image.load_img(filename, target_size=(224,224))
        # convert image to an array
        x = image.img_to_array(img)
        # expand image dimensions
        x = preprocess_input(x)
        x = np.expand_dims(x,axis=0)
        with graph.as_default():
            K.set_session(session)
            rs = model.predict(x)
            print(rs)  
        rs[0][0]
        rs[0][1]

        if rs[0][0] >= 0.9:
            result = "This lung CT image is NOT tumorous."
        else:
            result = "Warning! This lung CT image is tumorous."
    except:
        result = "Uploading error! Please upload image file..."

    return render(
        request,
        'app/analysis.html',
        {
            'title':'Result Page',
            'result':result,
            'year':datetime.now().year,
        }
    )

def contact(request):
    """Renders the contact page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/contact.html',
        {
            'title':'Contact',
            'message':'Your contact page.',
            'year':datetime.now().year,
        }
    )

def about(request):
    """Renders the about page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/about.html',
        {
            'title':'About',
            'message':'Application description page.',
            'year':datetime.now().year,
        }
    )

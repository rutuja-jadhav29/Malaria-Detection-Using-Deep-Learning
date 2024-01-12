from flask import Flask, render_template, request
import numpy as np
import re
from skimage import transform
import tensorflow as tf
from tensorflow.keras.models import Sequential

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

app = Flask(__name__)
loaded_model = load_model("models/model_malaria.h5")
#loaded_model._make_predict_function()
#graph = tf.get_default_graph
g = tf.Graph()
with g.as_default():
  # Define operations and tensors in `g`.
  c = tf.constant(30.0)
  assert c.graph is g
tf.compat.v1.get_default_graph()

def ValuePredictor(np_arr):   
    global graph
    with graph.as_default():
        result = loaded_model.predict(np_arr)
    return result[0]

def image_preprocess(img):
  new_shape = (50, 50, 3)
  img = image.load_img(img)
  image_array = image.img_to_array(img)
  image_array = transform.resize(image_array, new_shape, anti_aliasing = True)
  image_array /= 255
  image_array = np.expand_dims(image_array, axis = 0)
  return image_array

@app.route('/')
def home():
  return render_template("home.html")
@app.route('/result', methods = ['POST'])
def result():
    prediction=''
    if request.method == 'POST':
        img = request.files['pic']
        img_arr = image_preprocess(img)
        result = ValuePredictor(img_arr)
        print("result from model", result) 
        result = int(np.argmax(result))
        print("result actual", result) 
        if result==0:
            prediction='This cell is most likely to be Not Infected with Malarial Parasite.'
        else:
            prediction='This cell is most likely to be Infected with Malarial Parasite.'
        print(prediction)
        return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
  app.run()

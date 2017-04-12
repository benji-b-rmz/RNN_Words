# Benjamin Ramirez April 11, 2017
# WebApp for generating a mini-story from Recurrent Neural Networks

import numpy as np
from flask import Flask, render_template
import tensorflow as tf

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

app.run()
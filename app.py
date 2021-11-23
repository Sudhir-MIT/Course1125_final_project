from flask import Flask, render_template, request
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np 
import os 

threshold = 0.23 

#Load ML model here 
model_name = "saved_model/my_model.h5"
#Load model 
m = tf.keras.models.load_model(model_name,custom_objects={'KerasLayer':hub.KerasLayer})

app = Flask(__name__)

IMAGE_FOLDER = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER




@app.route('/', methods=['GET'])

def hello_world():
	return render_template('index.html')


@app.route('/', methods=['POST'])

def predict(): 
	imageFile = request.files['imageFile']
	image_path = None 

	if not imageFile:
		message = "Please select image!" 
		return render_template('index.html',user_image = image_path)

	image_path = os.path.join(app.config['UPLOAD_FOLDER'] , imageFile.filename) 
	imageFile.save(image_path)

	#Preprocess image 
	img = tf.io.read_file(image_path)
	img = tf.image.decode_jpeg(img, channels=3)
		
	img = tf.image.convert_image_dtype(img, tf.float32)
	# resize the image to the desired size.
	img = tf.image.resize(img, [299, 299])

	img = tf.reshape(img, [-1, 299, 299, 3])

	y_pred = m.predict(img)

	if y_pred >= threshold:
		message= 'Prediction: Malignant'
	else:
		message= 'Prediction: Benign' 



	return render_template('index.html', message=message, user_image = image_path)

if __name__ == '__main__':
	app.run(port = 3000, debug = True)
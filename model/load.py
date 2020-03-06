import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=FutureWarning)
	import tensorflow
	from tensorflow.keras.models import load_model


def init():
	# json_file = open('model/model.json','r')
	# loaded_model_json = json_file.read()
	# json_file.close()
	# loaded_model = model_from_json(loaded_model_json)
	# #load weights into new model
	# loaded_model.load_weights("model/model.h5")
	# print("Loaded Model from disk")
	#compile and evaluate loaded model
	loaded_model = load_model('model/digitClassifier.h5')
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	return loaded_model

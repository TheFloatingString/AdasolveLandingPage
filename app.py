# Framework
from flask import Flask, render_template, request
from werkzeug import secure_filename

# Data processing
import json
import pandas as pd
import numpy as np

# Machine leraning
from keras.models import Sequential, load_model
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

app = Flask(__name__)

#Variables
cancer_dict = {
	'Breast': 0,
	'Colorectum': 0,
	'Esophagus': 0,
	'Liver': 0,
	'Lung': 0,
	'Ovary': 0,
	'Pancreas': 0,
	'Stomach': 0
}

scaler_filename = "models/stephen_scaler.sav"
scaler = joblib.load(scaler_filename)

@app.route('/')
def index():
    return render_template("main.html")

@app.route("/form")
def form():
    return render_template("form.html")

@app.route("/analyze", methods=['GET', 'POST'])
def analyze():
	if request.method == 'POST':
		f = request.files['file']

		data = json.load(f)
		df = pd.DataFrame(data, index = [0])
		row_data = np.array(df, dtype=float)

		row_data = scaler.transform(row_data)

		nn_model = load_model("models/Detection.h5")
		nn_model_1 = load_model("models/multi-class-model.h5")

		predictions = nn_model.predict(row_data)

		result = ''
		percentage = 0
        
		for pred in predictions:
			if pred[0] > pred[1]:
				result = 'positive'
				percentage = str(round(pred[0] * 100, 2)) + '%'
			elif pred[1] > pred[0]:
				result = 'negative'
				percentage = str(round(pred[1] * 100, 2)) + '%'

		if result == 'positive':
			localization = nn_model_1.predict(row_data)

			for prediction in localization:
				prediction = list(prediction)
				del prediction[5]
				for cancer, score in cancer_dict.items():
					cancer_dict[cancer] = str(round(prediction[0] * 100, 2)) + '%'
					del prediction[0]

			return render_template("results.html", detection=result, percentage=percentage, localization=cancer_dict)

		else: return render_template("results.html", detection=result, percentage=percentage, localization={})

if __name__ == '__main__':
	app.run(debug=True)

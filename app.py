# Framework
from flask import Flask, render_template, request
from werkzeug import secure_filename

# Data processing
import json
import pandas as pd

# Machine leraning
from keras.models import Sequential, load_model

app = Flask(__name__)

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
        df = pd.DataFrame(data, index=[0])
        row_data = df.values

        nn_model = load_model("models/multi-class-model-prob.h5")
        prediction = nn_model.predict(row_data)
        prediction = prediction[0]

        return render_template("results.html",
        cancer_1=prediction[0]*100,
        cancer_2=prediction[1]*100,
        cancer_3=prediction[2]*100,
        cancer_4=prediction[3]*100,
        cancer_5=prediction[4]*100,
        cancer_6=prediction[5]*100,
        cancer_7=prediction[6]*100,
        cancer_8=prediction[7]*100,
        cancer_9=prediction[8]*100)

    else:
        return "An error occured!"

if __name__ == '__main__':
    app.run(debug=True)

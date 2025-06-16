from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('cancer_model.pkl', 'rb'))
app = Flask(__name__)




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = request.form['feature']
        features_list = features.split(',')
        
        # f = np.asarray(features_list).astype(np.float32)
        # features_list = f.tolist()
        
        final_features = np.asarray(features_list, dtype=np.float32).reshape(1, -1)
        prediction = model.predict(final_features)
        output = ["Cancerous" if prediction[0] == 1 else "Non-Cancerous"]
        # print(prediction[0])
    # output = prediction[0]  
    return render_template("index.html",message=output[0])
    # return render_template("index.html",message =f'Predicted class: {output}', prediction_text='The predicted class is: {}'.format(output))



if __name__ == '__main__':
    app.run(debug=True)
    
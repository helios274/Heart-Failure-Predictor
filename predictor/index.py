import pickle
import numpy as np

features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
            'ejection_fraction', 'high_blood_pressure', 'platelets',
            'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']

with open('predictor/models/model_gbc_v1.pkl', 'rb') as file:
    model = pickle.load(file)


def make_prediction(data):

    input_data = [data.get(f) for f in features]

    input_data = np.array(input_data).reshape(1, -1)

    heart_failure_prob = model.predict_proba(input_data)[0].tolist()[1]
    heart_failure_prob = round(heart_failure_prob*100, 2)

    return heart_failure_prob

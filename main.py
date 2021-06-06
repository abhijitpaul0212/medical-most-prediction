from flask import Flask, request, jsonify
from model_files.ml_model import predict_cost
import joblib

app = Flask("medical_cost_prediction")

@app.route('/', methods=['POST'])
def predict():
    input_data = request.get_json() 
    
    model = joblib.load("./model_files/medical_cost_prediction_model.pkl")
    # with open('./model_files/medical_cost_prediction_model.pkl', 'rb') as f_in:
    #     model = pickle.load(f_in)
    #     f_in.close()
    
    predictions = predict_cost(input_data, model)
    response = {
        'medical_cost_predictions': list(predictions)
    }    
    return jsonify(response)

# @app.route('/', methods=['GET'])
# def ping():
#     return "Pinging model application"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
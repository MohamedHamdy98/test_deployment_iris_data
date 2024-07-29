from flask import Flask, request, jsonify
import pickle


# Load the serialized model
with open('iris_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize Flask application
app = Flask(__name__)

stored_features  = None

@app.route('/')
def home():
    return 'iris dataset'

# Define a route for prediction
@app.post('/predict')
def predict():
    global stored_features
    try:
        # Get data from request
        data = request.json

        features = data['f']
        stored_features  = features
        # Perform prediction using the loaded model
        predictions = model.predict([features]) 
        
        # Prepare and return the response as JSON
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.get("/getData")
def get_predictions():
    global stored_features 
    try:
        if stored_features  is None:
            return jsonify({'error': 'No features available. Please make a POST request to /predict first.'}), 400
        # Make a POST request to the /predict endpoint
        import requests
        response = requests.post('http://127.0.0.1:5000/predict', json={'f': stored_features})
        
        # Check if the response is successful
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Failed to get predictions'}), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


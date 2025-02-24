from flask import Blueprint, jsonify
from .preprocess import preprocess_data
from .train_model import train_model, make_prediction

sales_prediction_bp = Blueprint('sales_prediction', __name__)

@sales_prediction_bp.route('/preprocess', methods=['GET'])
def preprocess():
    preprocess_data()
    return jsonify({'message': 'Preprocessing completed.'})

@sales_prediction_bp.route('/train', methods=['GET'])
def train():
    train_model()
    return jsonify({'message': 'Model training completed.'})

@sales_prediction_bp.route('/predict', methods=['GET'])
def predict():
    prediction = make_prediction()
    return jsonify({'prediction': prediction})

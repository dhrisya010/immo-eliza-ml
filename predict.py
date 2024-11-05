

import pickle
import numpy as np
from trian import testing_data

file = 'D:\\BeCode\\Projects\\immo-eliza-ml\\test_data.csv'

def predict_price(input_data):
    # Load the trained model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    
   
if __name__ == "__main__":
    # Example input data for a new house
    input_data = {
        'construction_year': 1999,
        'nbr_bedrooms': 3,  
        'total_area_sqm': 850,
        'terrace_sqm': 20  
    }
    
    # Predict the price
    predicted_price = testing_data(file)
    print(f"Predicted Price for the House: €{predicted_price}")










































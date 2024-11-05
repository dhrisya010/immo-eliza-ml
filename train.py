import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt


file_path = "D:\\BeCode\\Projects\\immo-eliza-ml\\properties.csv"

def clean_data(file_path):
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Remove duplicates
    df = df.drop_duplicates()

    # Remove white spaces 
    df = df.apply(lambda x: x.str.strip().str.replace('\s+', ' ', regex=True) if x.dtype == "object" else x)

    # Ensure 'price' column contains only numeric values, setting non-numeric entries to NaN
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # List of columns you want to drop
    columns_to_drop = ['id', 'locality','province','subproperty_type','nbr_frontages', 'cadastral_income', 'epc', 'surface_land_sqm','fl_terrace','fl_garden','latitude','longitude']
    
    # Drop columns if they exist in the DataFrame
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
   
    #  Save the cleaned data to a new file if output_path is provided
    cleaned_file_path = 'output_file.csv'
    df.to_csv(cleaned_file_path, index=False) 
    #df.to_csv('output_file', index=False)
    print(f"Cleaned data saved.")

    return df
clean_data(file_path)



file = 'D:\\BeCode\\Projects\\immo-eliza-ml\\output_file.csv'
def data_preprocessing(file):
    df = pd.read_csv(file)
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    # Impute missing values for numerical columns
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])
    # Impute missing values for categorical columns
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        df[col] = df[col].fillna('Unknown')

    encoded_df = df[['property_type','region','equipped_kitchen','state_building','heating_type']]
    # Create encoder object
    encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
    new_columns = encoder.fit_transform(encoded_df)
    df = pd.concat([df, new_columns], axis=1).drop(columns=['property_type','region','equipped_kitchen','state_building','heating_type'])
    
    df_features = df.drop(columns=['price'])
    columns = df_features.columns
    # Initialize the scaler
    scaler = StandardScaler()
    # Apply the scaler to the numeric features
    df[columns] = scaler.fit_transform(df[columns])

    # Save the file 
    imputed_file_path = 'properties_imputed.csv'
    df.to_csv(imputed_file_path, index=False)
    print("Preprocessing  file is created")

    # Split the dataset to train and test
    df.sample(frac =0.8).to_csv('train_data.csv', index=False)
    df.sample(frac=0.2).to_csv('test_data.csv', index=False)
    

data_preprocessing(file)



file = 'D:\\BeCode\\Projects\\immo-eliza-ml\\train_data.csv'
def training_data(file):
    df= pd.read_csv(file)
    X= df.drop(['price'],axis = 1)
    y= df['price']
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size= 0.2, random_state= 42)
    # Identify categorical features and one-hot encode them
    categorical_features = X_train.select_dtypes(include=['object']).columns
    X_train = pd.get_dummies(X_train, columns=categorical_features)
    X_test = pd.get_dummies(X_test, columns=categorical_features)
    
    # Define and train the model
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    print("Train Score :", train_score)

    # Pickle the model
    with open('model.pkl', 'wb') as file: #writing bina file for module
        pickle.dump(model, file)


    print("Pickle file is created")

 # Call the function 
training_data(file)



file = 'D:\\BeCode\\Projects\\immo-eliza-ml\\test_data.csv'
def testing_data(file):
    df= pd.read_csv(file)
    X= df.drop(['price'],axis = 1)
    y= df['price']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Load the trained model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Identify categorical features and one-hot encode them
    categorical_features = X_test.select_dtypes(include=['object']).columns
    X_test = pd.get_dummies(X_test, columns=categorical_features)
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Model Evaluation on Test Data:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2): {r2}")
    return y_pred
    




























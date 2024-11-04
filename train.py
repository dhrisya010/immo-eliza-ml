import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
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




def data_preprocessing():
    df = pd.read_csv('D:\\BeCode\\Projects\\immo-eliza-ml\\output_file.csv')
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
    

data_preprocessing()


def training_data():
    df= pd.read_csv('D:\\BeCode\\Projects\\immo-eliza-ml\\properties_imputed.csv')
    X= df.drop(['price'],axis = 1)
    y= df['price']
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size= 0.2, random_state= 42)
    # Identify categorical features and one-hot encode them
    categorical_features = X_train.select_dtypes(include=['object']).columns
    X_train = pd.get_dummies(X_train, columns=categorical_features)
    X_test = pd.get_dummies(X_test, columns=categorical_features)
    
    # Align columns to make sure both train and test data have the same structure
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    # Define and train the model
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    # Pickle the model
    with open('model.pkl', 'wb') as file: #writing bina file for module
        pickle.dump(model, file)


    print("Pickle file is created")


training_data()

def testing_data():
    df= pd.read_csv('D:\\BeCode\\Projects\\immo-eliza-ml\\properties_imputed.csv')
    df.head(20)
    X= df.drop(['price'],axis = 1)
    y= df['price']
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size= 0.2, random_state= 42)
    # Identify categorical features and one-hot encode them
    categorical_features = X_train.select_dtypes(include=['object']).columns
    X_train = pd.get_dummies(X_train, columns=categorical_features)
    X_test = pd.get_dummies(X_test, columns=categorical_features)
    
    # Align columns to make sure both train and test data have the same structure
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    # Define and train the model
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    # Pickle the model
    with open('model.pkl', 'wb') as file: #writing bina file for module
        pickle.dump(model, file)


    print("Pickle file is created")


testing_data()



# importing pandas package 
import pandas as pd 
  
# making data frame from csv file  
data = pd.read_csv("D:\\BeCode\\Projects\\immo-eliza-ml\\properties_imputed.csv") 
  
# generating one row  
rows = data.sample(frac =.08)





















'''



def standardize(file_path, df):
    df_features = df.drop(columns=['price'])
    columns = df_features.columns
    # Initialize the scaler
    scaler = StandardScaler()
    # Apply the scaler to the numeric features
    df[columns] = scaler.fit_transform(df[columns])
    print(df.head())


def preprocess_data():
    imputed_columns()
    encoding_columns()
    standardize()




def split_data(data, target_column, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train):
    """Train a regression model."""
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return RMSE."""
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse

def save_model(model, filepath):
    """Save the trained model to a file."""
    joblib.dump(model, filepath)

'''
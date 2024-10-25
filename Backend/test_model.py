import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# Define mapping dictionaries
medical_condition_mapping = {
    'asthma': 'Asthma',
    'hypertension': 'Hypertension',
    'high blood pressure': 'Hypertension',
    'diabetes': 'Diabetes',
    'thyroid': 'Thyroid',
    'none': 'None',
}

goal_mapping = {
    'weight loss': 'Weight Loss',
    'weight gain': 'Weight Gain',
    'muscle building': 'Build Muscle',
    'build muscle': 'Build Muscle',
    'increase stamina': 'Increase Stamina',
    'fitness': 'Maintain Fitness',
    'maintain fitness': 'Maintain Fitness',
}

diet_preference_mapping = {
    'veg': 'Veg',
    'vegetarian': 'Veg',
    'non-veg': 'Non-Veg',
    'non vegetarian': 'Non-Veg',
    'both': 'Both',
}

# Define valid options for error handling
valid_medical_conditions = ['asthma', 'hypertension', 'high blood pressure', 'diabetes', 'thyroid', 'none']
valid_goals = ['weight loss', 'weight gain', 'muscle building', 'build muscle', 'increase stamina', 'fitness', 'maintain fitness']
valid_diet_preferences = ['veg', 'vegetarian', 'non-veg', 'non vegetarian', 'both']

# Function to map user input to the predefined categories
def map_user_input(medical_condition, goal, diet_preference):
    medical_condition = medical_condition_mapping.get(medical_condition.lower(), 'None')
    goal = goal_mapping.get(goal.lower(), 'Maintain Fitness')  # Default to 'Maintain Fitness'
    diet_preference = diet_preference_mapping.get(diet_preference.lower(), 'Veg')  # Default to 'Veg'
    return medical_condition, goal, diet_preference

# Function to validate user inputs
def validate_input(user_input, valid_options, input_name):
    if user_input.lower() not in valid_options:
        raise ValueError(f"Invalid {input_name}: '{user_input}'. Valid options are: {', '.join(valid_options)}.")

# Function to load the model and encoder
def load_model_and_encoder(filepath):
    with open(filepath, 'rb') as file:
        model_data = pickle.load(file)
    return model_data['model'], model_data['encoder']

# Function to make predictions
def make_prediction(model, encoder, weight, height, medical_condition, goal, diet_preference):
    sample_data = {
        'Weight_kg': [weight],
        'Height_cm': [height],
        'Medical_Condition': [medical_condition],
        'Goal': [goal],
        'Diet_Preference': [diet_preference]
    }

    input_df = pd.DataFrame(sample_data)
    input_df['Weight_kg'] = input_df['Weight_kg'].astype(float)
    input_df['Height_cm'] = input_df['Height_cm'].astype(float)

    # Handle unknown categories by ignoring them
    input_encoded = encoder.transform(input_df[['Medical_Condition', 'Goal', 'Diet_Preference']])
    encoded_columns = encoder.get_feature_names_out(['Medical_Condition', 'Goal', 'Diet_Preference'])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoded_columns)

    input_final = pd.concat([input_encoded_df, input_df[['Weight_kg', 'Height_cm']].reset_index(drop=True)], axis=1)

    predictions = model.predict(input_final)
    return predictions[0]

# Function to train the model
def train_model(data_filepath, model_save_path):
    # Load and preprocess data
    data = pd.read_csv(data_filepath, encoding='ISO-8859-1')
    data.columns = data.columns.str.strip()  # Clean column names
    data['Weight_kg'] = data['Weight_kg'].str.replace('kg', '').astype(float)
    data['Height_cm'] = data['Height_cm'].str.replace('cm', '').astype(float)
    data['Medical_Condition'] = data['Medical_Condition'].fillna('None')

    # Ensure consistency in categorical values
    data['Medical_Condition'] = data['Medical_Condition'].map(medical_condition_mapping)
    data['Goal'] = data['Goal'].map(goal_mapping)
    data['Diet_Preference'] = data['Diet_Preference'].map(diet_preference_mapping)

    # Define features and target variable
    X = data[['Weight_kg', 'Height_cm', 'Medical_Condition', 'Goal', 'Diet_Preference']]
    y = data['Diet_Plan']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Encode categorical variables, allowing unknown categories
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X_train[['Medical_Condition', 'Goal', 'Diet_Preference']]).toarray()

    # Replace original columns with encoded ones
    X_train_encoded = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['Medical_Condition', 'Goal', 'Diet_Preference']))
    X_train_encoded['Weight_kg'] = X_train['Weight_kg'].reset_index(drop=True)
    X_train_encoded['Height_cm'] = X_train['Height_cm'].reset_index(drop=True)

    # Create and train the model
    model = RandomForestClassifier()
    model.fit(X_train_encoded, y_train)

    # Save the model and encoder
    model_data = {
        'model': model,
        'encoder': encoder
    }

    with open(model_save_path, 'wb') as file:
        pickle.dump(model_data, file)

    print("Model and encoder saved successfully!")

# Main execution block
if __name__ == "__main__":
    # Specify the file paths
    model_save_path = r"S:\Cook_Smart_website\Backend\diet_model.pkl"
    data_filepath = r"C:\ProgramData\MySQL\MySQL Server 9.0\Uploads\final_cookbotdb.csv"

    # Uncomment to train the model
    # train_model(data_filepath, model_save_path)

    # Load the model and encoder
    model, encoder = load_model_and_encoder(model_save_path)

    # Take user input from command line with error handling
    try:
        while True:
            try:
                weight = float(input("Enter your weight in kg: "))
                break  # Exit the loop if input is valid
            except ValueError:
                print("Please enter a valid weight in kg (numerical value).")

        while True:
            try:
                height = float(input("Enter your height in cm: "))
                break  # Exit the loop if input is valid
            except ValueError:
                print("Please enter a valid height in cm (numerical value).")

        user_medical_condition = input("Enter your medical condition (Diabetes, Hypertension, Asthma, Thyroid, None): ")
        validate_input(user_medical_condition, valid_medical_conditions, "medical condition")

        user_goal = input("Enter your goal (Weight Loss, Weight Gain, Build Muscle, Increase Stamina, Maintain Fitness): ")
        validate_input(user_goal, valid_goals, "goal")

        user_diet_preference = input("Enter your diet preference (Veg, Non-Veg, Both): ")
        validate_input(user_diet_preference, valid_diet_preferences, "diet preference")

        # Map user input to the model's expected categories
        mapped_medical_condition, mapped_goal, mapped_diet_preference = map_user_input(
            user_medical_condition, user_goal, user_diet_preference
        )

        # Make predictions
        predicted_diet_plan = make_prediction(model, encoder, weight, height, mapped_medical_condition, mapped_goal, mapped_diet_preference)

        # Output the prediction
        print("Predicted Diet Plan:", predicted_diet_plan)

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

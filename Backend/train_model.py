import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import pickle
import chardet

# Step 1: Detect the encoding of the CSV file
with open(r"C:\ProgramData\MySQL\MySQL Server 9.0\Uploads\final_cookbotdb.csv", 'rb') as f:
    result = chardet.detect(f.read())
print("Detected encoding:", result['encoding'])

# Step 2: Load the data with the detected encoding
data = pd.read_csv(r"C:\ProgramData\MySQL\MySQL Server 9.0\Uploads\final_cookbotdb.csv", encoding=result['encoding'])

# Step 3: Strip whitespace from column names
data.columns = data.columns.str.strip()

# Step 4: Check if required columns are present before proceeding
required_columns = ['Weight_kg', 'Height_cm', 'Medical_Condition', 'Goal', 'Diet_Preference', 'Diet_Plan']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    raise KeyError(f"Missing columns in the DataFrame: {missing_columns}")

# Step 5: Preprocessing the data
# Clean up 'Weight_kg' and 'Height_cm' by removing units and converting to floats
data['Weight_kg'] = data['Weight_kg'].str.replace('kg', '').astype(float)
data['Height_cm'] = data['Height_cm'].str.replace('cm', '').astype(float)
data['Medical_Condition'] = data['Medical_Condition'].fillna('None')  # Fill missing medical conditions with 'None'

# Define the features (X) and the target variable (y)
X = data[required_columns[:-1]]  # All except 'Diet_Plan' are features
y = data['Diet_Plan']  # 'Diet_Plan' is the target

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Encode categorical variables using OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Set handle_unknown to 'ignore'
X_train_encoded = encoder.fit_transform(X_train[['Medical_Condition', 'Goal', 'Diet_Preference']])

# Get feature names for encoded variables
encoded_columns = encoder.get_feature_names_out(['Medical_Condition', 'Goal', 'Diet_Preference'])
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_columns)

# Combine the encoded categorical variables with the numeric ones (Weight and Height)
X_train_final = pd.concat([X_train_encoded_df, X_train[['Weight_kg', 'Height_cm']].reset_index(drop=True)], axis=1)

# Step 8: Train the model using RandomForestClassifier
model = RandomForestClassifier(random_state=42)  # Added random_state for reproducibility
model.fit(X_train_final, y_train)

# Step 9: Save the model and encoder using pickle
model_data = {
    'model': model,
    'encoder': encoder
}

with open(r"S:\Cook_Smart_website\Backend\diet_model.pkl", 'wb') as file:
    pickle.dump(model_data, file)

print("Model and encoder saved successfully!")


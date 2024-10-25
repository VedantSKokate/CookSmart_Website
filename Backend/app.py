import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = 'ab4602bcb116f048f4e7fc0069d3ad3a'

# MySQL connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Vsk@26",
    database="cook_smart_db"
)

# Load trained model and encoder
with open('diet_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    encoder = model_data['encoder']

# Load and clean recipe dataset
file_path = r"S:\Cook_Smart_website\recipe_final.csv"
recipe_df = pd.read_csv(file_path).dropna()
recipe_df.columns = recipe_df.columns.str.strip()

# Ensure numeric columns are correctly parsed
for col in ['Calories', 'Protein_gm', 'Fiber_gm']:
    if col in recipe_df.columns:
        recipe_df[col] = pd.to_numeric(recipe_df[col], errors='coerce')

# Mappings for user inputs
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
    'increase stamina': 'Increase Stamina',
    'fitness': 'Maintain Fitness',
}

diet_preference_mapping = {
    'veg': 'Veg',
    'vegetarian': 'Veg',
    'non-veg': 'Non-Veg',
    'non vegetarian': 'Non-Veg',
    'both': 'Both',
}

# Map user input to valid values
def map_user_input(medical_condition, goal, diet_preference):
    return (
        medical_condition_mapping.get(medical_condition.lower(), 'None'),
        goal_mapping.get(goal.lower(), 'Maintain Fitness'),
        diet_preference_mapping.get(diet_preference.lower(), 'Veg')
    )

# Predict diet plan
def predict_diet_plan(user_data):
    try:
        weight = float(user_data.get('weight'))
        height = float(user_data.get('height'))
        mc, g, dp = map_user_input(
            user_data.get('medical_condition'),
            user_data.get('goal'),
            user_data.get('diet_preference')
        )

        input_features = pd.DataFrame({
            'Weight_kg': [weight],
            'Height_cm': [height],
            'Medical_Condition': [mc],
            'Goal': [g],
            'Diet_Preference': [dp]
        })

        encoded_input = encoder.transform(input_features[['Medical_Condition', 'Goal', 'Diet_Preference']])
        encoded_columns = encoder.get_feature_names_out(['Medical_Condition', 'Goal', 'Diet_Preference'])
        encoded_df = pd.DataFrame(encoded_input, columns=encoded_columns)

        final_input = pd.concat([encoded_df, input_features[['Weight_kg', 'Height_cm']].reset_index(drop=True)], axis=1)

        prediction = model.predict(final_input) if hasattr(model, 'predict') else "Invalid model"
        return f"Diet Plan for {user_data.get('goal')}: {prediction[0]}"
    except Exception as e:
        return str(e)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recipe-suggestion', methods=['GET', 'POST'])
def recipe_suggestion():
    recipes = []
    message = ""
    user_specifications = {}

    if request.method == 'POST':
        ingredients = request.form.get('ingredients', '').lower().split(',')
        protein = request.form.get('protein')
        fiber = request.form.get('fiber')
        calories = request.form.get('calories')
        diet_preference = request.form.get('diet_type').lower()  # Ensure lower case comparison

        user_specifications = {
            "Ingredients": request.form.get('ingredients'),
            "Max Protein": protein,
            "Max Fiber": fiber,
            "Max Calories": calories,
            "Diet Preference": diet_preference
        }

        # Start with all recipes
        filtered_recipes = recipe_df.copy()

        # Filter by ingredients if provided
        if ingredients[0].strip():
            ingredients = [ing.strip() for ing in ingredients if ing]
            filtered_recipes = filtered_recipes[
                filtered_recipes['Ingredients'].apply(lambda x: any(ing in x.lower() for ing in ingredients))
            ]

        # Filter by diet type
        if diet_preference in ['veg', 'non-veg']:
            filtered_recipes = filtered_recipes[
                filtered_recipes['Veg/Non-Veg'].str.strip().str.lower() == diet_preference      #case handling solved issue
            ]

        # Apply numeric filters
        if protein and protein.isnumeric():
            filtered_recipes = filtered_recipes[filtered_recipes['Protein_gm'] <= int(protein)]
        if fiber and fiber.isnumeric():
            filtered_recipes = filtered_recipes[filtered_recipes['Fiber_gm'] <= int(fiber)]
        if calories and calories.isnumeric():
            filtered_recipes = filtered_recipes[filtered_recipes['Calories'] <= int(calories)]

        # Convert to list of dictionaries for display
        recipes = filtered_recipes.to_dict(orient='records')

        if not recipes:
            message = "No recipes found based on your criteria."

    return render_template('recipe_suggestion.html', recipes=recipes, message=message, user_specifications=user_specifications)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
            (username, email, password)
        )
        db.commit()
        cursor.close()
        flash('Signup successful!', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('home'))

        flash('Invalid email or password.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_data = request.get_json()
    diet_plan = predict_diet_plan(user_data)
    return jsonify({'diet_plan': diet_plan})

if __name__ == '__main__':
    app.run(debug=True)

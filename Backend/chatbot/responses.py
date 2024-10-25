# responses.py

from nlp import normalize_user_data

def handle_user_input(user_input):
    # Assuming user_input is a dictionary with all the required fields
    weight_input = user_input.get("weight")
    height_input = user_input.get("height")
    medical_condition_input = user_input.get("medical_condition")
    goal_input = user_input.get("goal")
    diet_preference_input = user_input.get("diet_preference")
    
    normalized_data = normalize_user_data(weight_input, height_input, medical_condition_input, goal_input, diet_preference_input)
    
    # Use normalized_data for further processing, e.g., querying the database
    return normalized_data

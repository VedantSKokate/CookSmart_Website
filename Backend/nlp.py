def parse_user_input(user_input):
    # Assume user_input is in a structured format like "Weight: 100kg, Height: 165cm, Condition: Diabetes, Goal: Weight loss, Preference: Veg"
    details = {}
    try:
        for detail in user_input.split(","):
            key, value = detail.split(":")
            details[key.strip()] = value.strip()
    except ValueError:
        return "Invalid input format. Please provide input in the format: Weight: x, Height: y, Condition: z, Goal: a, Preference: b"

    return details

import pandas as pd
from flask import Flask, request, jsonify
import pickle
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

model_path = '/Users/milarize/Documents/foodpet/foodforpet.pkl'

# Load data
with open(model_path, 'rb') as file:
    contents = pickle.load(file)
print(f"Loaded contents: {contents}")
print(f"Type of label_encoders: {type(contents[1])}")
print(f"Label encoders: {contents[1]}")
# Check number and type of objects loaded
print(f"Number of objects in the pickle file: {len(contents)}")
for i, obj in enumerate(contents):
    print(f"Object {i}: {type(obj)}")

if len(contents) >= 2:
    product_name_model, label_encoders = contents[0], contents[1]  # Assuming first two are models
    other_models = contents[2:]  # Get remaining models if any
else:
    raise ValueError("Unexpected number of objects in the pickle file.")

# Check if label_encoders is a dictionary
if not isinstance(label_encoders, dict):
    raise ValueError("Label encoders should be a dictionary.")

# Check for required keys
required_keys = ['ชนิดของอาหาร', 'ประเภทอาหาร', 'ประเภทสัตว์', 'ชื่อของผลิตภัณฑ์']
for key in required_keys:
    if key not in label_encoders:
        raise ValueError(f"Missing required label encoder for key: {key}")

# Function to recommend pet food
def recommend_pet_food(weight, age_years, age_months, food_type, food_category, pet_type, budget=None):
    try:
        # Encode the input values
        food_type_encoded = label_encoders['ชนิดของอาหาร'].transform([food_type])[0]
        food_category_encoded = label_encoders['ประเภทอาหาร'].transform([food_category])[0]
        animal_type_encoded = label_encoders['ประเภทสัตว์'].transform([pet_type])[0]

        # Create DataFrame for prediction input
        input_data = pd.DataFrame([[animal_type_encoded, weight, age_years, age_months, food_type_encoded, food_category_encoded, budget if budget else 0]],
                                  columns=['ประเภทสัตว์', 'น้ำหนัก (กก.)', 'อายุ (ปี)', 'อายุ (เดือน)', 'ชนิดของอาหาร', 'ประเภทอาหาร', 'งบประมาณ (บาท)'])

        # Use model to predict product name
        product_name_encoded = product_name_model.predict(input_data)[0]
        
    except KeyError as e:
        return f"Key error: {str(e)}. Check if the input values are valid."
    except Exception as e:
        return f"An error occurred while encoding input values: {str(e)}"

    # Retrieve matching products based on predicted name and budget
    result = data[(data['ชื่อของผลิตภัณฑ์'] == product_name_encoded) &
                  (data['ประเภทอาหาร'] == food_category_encoded)]
    if budget:
        result = result[result['งบประมาณ (บาท)'] <= budget]

    # Expand filter criteria if no results found
    if result.empty:
        result = data[(data['ประเภทอาหาร'] == food_category_encoded) &
                      (data['ชนิดของอาหาร'] == food_type_encoded)]
        if budget:
            result = result[result['งบประมาณ (บาท)'] <= budget]

    # If still no results, return message
    if result.empty:
        return "ไม่มีผลิตภัณฑ์ที่ตรงกับเงื่อนไขที่กำหนด"

    # Prepare recommendations
    recommendations = []
    for idx, row in result.iterrows():
        food_category_from_data = label_encoders['ประเภทอาหาร'].inverse_transform([row['ประเภทอาหาร']])[0]
        food_type_from_data = label_encoders['ชนิดของอาหาร'].inverse_transform([row['ชนิดของอาหาร']])[0]
        animal_type_from_data = label_encoders['ประเภทสัตว์'].inverse_transform([row['ประเภทสัตว์']])[0]

        recommendations.append({
            "ผลิตภัณฑ์ที่แนะนำ": label_encoders['ชื่อของผลิตภัณฑ์'].inverse_transform([row['ชื่อของผลิตภัณฑ์']])[0],
            "ประเภทสัตว์": animal_type_from_data,
            "ชนิดของอาหาร": food_type_from_data,
            "ประเภทอาหาร": food_category_from_data,
            "ราคา": row['งบประมาณ (บาท)'],
            "ปริมาณอาหาร (กก.)": row['ปริมาณอาหาร (กก.)'],
            "สารอาหาร": {
                "Protein": row['สัดส่วนของสารอาหารต่อ 100 กรัม (Protein)'],
                "Fat": row['สัดส่วนของสารอาหารต่อ 100 กรัม (Fat)'],
                "Fiber": row['สัดส่วนของสารอาหารต่อ 100 กรัม (Fiber)'],
                "Calcium": row['สัดส่วนของสารอาหารต่อ 100 กรัม (Calcium)'],
                "Ash": row['สัดส่วนของสารอาหารต่อ 100 กรัม (Ash)'],
                "Phosphorus": row['สัดส่วนของสารอาหารต่อ 100 กรัม (Phosphorus)'],
                "Omega-6": row['สัดส่วนของสารอาหารต่อ 100 กรัม (Omega-6 fatty acids)'],
                "Moisture": row['สัดส่วนของสารอาหารต่อ 100 กรัม (Moisture)']
            }
        })

    # Filter unique products and return top 3
    unique_recommendations = {}
    for rec in recommendations:
        volume = rec["ปริมาณอาหาร (กก.)"]
        unique_recommendations[volume] = rec

    return list(unique_recommendations.values())[:3]

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()

    # Input validation
    required_fields = ['weight', 'age_years', 'age_months', 'food_type', 'food_category', 'pet_type', 'budget']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"{field} is required"}), 400

    # Call the recommendation function with validated inputs
    try:
        weight = float(data['weight'])
        age_years = int(data['age_years'])
        age_months = int(data['age_months'])
        food_type = data['food_type']
        food_category = data['food_category']
        pet_type = data['pet_type']
        budget = float(data['budget'])
    except ValueError as e:
        return jsonify({"error": "Invalid input type"}), 400

    result = recommend_pet_food(weight, age_years, age_months, food_type, food_category, pet_type, budget)
    return jsonify(result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)

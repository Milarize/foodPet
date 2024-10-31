from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

app = FastAPI()


file_path = 'foodpetAll.csv'
data = pd.read_csv(file_path)


data['งบประมาณ (บาท)'] = data['งบประมาณ (บาท)'].str.replace(',', '').astype(float)


label_encoders = {}
for column in ['ประเภทสัตว์', 'ชนิดของอาหาร', 'ประเภทอาหาร', 'ชื่อของผลิตภัณฑ์']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le


features = data[['ประเภทสัตว์', 'น้ำหนัก (กก.)', 'อายุ (ปี)', 'อายุ (เดือน)', 'ชนิดของอาหาร', 'ประเภทอาหาร', 'งบประมาณ (บาท)']]
target_product_name = data['ชื่อของผลิตภัณฑ์']
X_train, X_test, y_train, y_test = train_test_split(features, target_product_name, test_size=0.2, random_state=42)

product_name_model = DecisionTreeClassifier()
product_name_model.fit(X_train, y_train)


class PetInfo(BaseModel):
    weight: float
    age_years: int
    age_months: int
    food_type: str
    food_category: str
    animal_type: str
    budget: float = None


def recommend_dog_food(weight, age_years, age_months, food_type, food_category, animal_type, budget=None):

    try:
        food_type_encoded = label_encoders['ชนิดของอาหาร'].transform([food_type])[0]
    except KeyError:
        return {"error": f"Food type '{food_type}' not recognized."}

    try:
        food_category_encoded = label_encoders['ประเภทอาหาร'].transform([food_category])[0]
    except KeyError:
        return {"error": f"Food category '{food_category}' not recognized."}

    try:
        animal_type_encoded = label_encoders['ประเภทสัตว์'].transform([animal_type])[0]
    except KeyError:
        return {"error": f"Animal type '{animal_type}' not recognized."}

    input_data = pd.DataFrame([[animal_type_encoded, weight, age_years, age_months, food_type_encoded, food_category_encoded, budget if budget else 0]],
                              columns=['ประเภทสัตว์', 'น้ำหนัก (กก.)', 'อายุ (ปี)', 'อายุ (เดือน)', 'ชนิดของอาหาร', 'ประเภทอาหาร', 'งบประมาณ (บาท)'])

    product_name_encoded = product_name_model.predict(input_data)[0]

    if budget:
        result = data[(data['ชื่อของผลิตภัณฑ์'] == product_name_encoded) &
                      (data['ประเภทอาหาร'] == food_category_encoded) &
                      (data['งบประมาณ (บาท)'] <= budget)]
    else:
        result = data[data['ชื่อของผลิตภัณฑ์'] == product_name_encoded]

    if result.empty:
        result = data[(data['ประเภทอาหาร'] == food_category_encoded) &
                      (data['ชนิดของอาหาร'] == food_type_encoded)]
        if budget:
            result = result[result['งบประมาณ (บาท)'] <= budget]

    if result.empty:
        return []

    recommendations = []
    for idx, row in result.iterrows():
        recommendations.append({
            "ผลิตภัณฑ์ที่แนะนำ": label_encoders['ชื่อของผลิตภัณฑ์'].inverse_transform([row['ชื่อของผลิตภัณฑ์']])[0],
            "ประเภทสัตว์": label_encoders['ประเภทสัตว์'].inverse_transform([row['ประเภทสัตว์']])[0],
            "ชนิดของอาหาร": label_encoders['ชนิดของอาหาร'].inverse_transform([row['ชนิดของอาหาร']])[0],
            "ประเภทอาหาร": label_encoders['ประเภทอาหาร'].inverse_transform([row['ประเภทอาหาร']])[0],
            "ราคา": row['งบประมาณ (บาท)'],
            "ปริมาณอาหาร (กก.)": row['ปริมาณอาหาร (กก.)'],
        })

    return recommendations[:3]

@app.post("/recommend")
def recommend(pet_info: PetInfo):
    recommendations = recommend_dog_food(
        pet_info.weight,
        pet_info.age_years,
        pet_info.age_months,
        pet_info.food_type,
        pet_info.food_category,
        pet_info.animal_type,
        pet_info.budget
    )
    return {"recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

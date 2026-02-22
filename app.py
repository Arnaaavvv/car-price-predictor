from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("lasso_regression_model.pkl")
car_data = pd.read_csv('Cleaned_Data.csv')

FUEL_TYPE_MAP = {
    0: "Gasoline",
    1: "Diesel",
    2: "Hybrid",
    3: "Flex Fuel",
    4: "Plug-In Hybrid"
}

@app.route('/')
def index():
    brands = sorted(car_data['brand'].unique().tolist())
    car_models = sorted(car_data['model'].unique().tolist())
    years = sorted(car_data['model_year'].unique().tolist(), reverse=True)

    brand_model_map = car_data.groupby('brand')['model'].apply(list).to_dict()

    model_details_map = {}
    for m, group in car_data.groupby('model'):
        model_details_map[m] = {
            'engines': sorted(group['engine'].unique().tolist()),
            'transmissions': sorted(group['transmission'].unique().tolist()),
            'fuel_types': [
                {"value": int(ft), "label": FUEL_TYPE_MAP.get(int(ft), str(ft))}
                for ft in sorted(group['fuel_type'].unique().tolist())
            ]
        }

    accident_options = [
        {"value": 0, "label": "No Accidents"},
        {"value": 1, "label": "Has Accidents"}
    ]

    return render_template(
        'index.html',
        brands=brands,
        car_models=car_models,
        years=years,
        accident_options=accident_options,
        brand_model_map=brand_model_map,
        model_details_map=model_details_map
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        brand = request.form.get('brands')
        car_model = request.form.get('car_model')
        year = int(request.form.get('year'))
        milage = int(request.form.get('milage'))
        fuel_type = int(request.form.get('fuel_type'))
        engine = request.form.get('engine')
        transmission = request.form.get('transmission')
        accidents = int(request.form.get('accidents'))
    except (TypeError, ValueError) as e:
        return f"Invalid input: {str(e)}", 400

    prediction = model.predict(pd.DataFrame(
        [[brand, car_model, year, milage, fuel_type, engine, transmission, accidents]],
        columns=['brand', 'model', 'model_year', 'milage', 'fuel_type', 'engine', 'transmission', 'accident']
    ))

    price = float(np.round(prediction[0],3))
    price = max(0,price)

    return str(price)

if __name__ == "__main__":
    app.run(debug=True)
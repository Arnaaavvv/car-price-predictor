# 🚗 Car Price Predictor 💰

A machine learning-powered web application built with Flask that predicts used car prices based on key vehicle attributes. The model is trained on a dataset of 3,547 real used car listings spanning 48 brands and 1,686 models.

---

## Preview


| Field | Options |
|---|---|
| Brand | 48 brands (Ford, Toyota, BMW, etc.) |
| Model | 1,686 models (dynamically filtered by brand) |
| Year | 2000 – 2024 |
| Mileage | Any value in miles |
| Fuel Type | Gasoline, Diesel, Hybrid, Flex Fuel, Plug-In Hybrid |
| Engine | Filtered based on selected model |
| Transmission | Filtered based on selected model |
| Accident History | No Accidents / Has Accidents |

---

## 🧠 How It Works

1. User selects car details through a dynamic, cascading form
2. Form data is sent via AJAX to the Flask backend (`/predict` route)
3. The backend feeds the input into a trained **Lasso Regression** pipeline
4. The predicted price is formatted to 3 decimal places with commas, then returned and displayed

### Model Details

| Property | Value |
|---|---|
| Algorithm | Lasso Regression (`alpha=0.1`) |
| Pipeline | `ColumnTransformer` + `OneHotEncoder` + `Lasso` |
| Training Data | 3,547 used car listings |
| Features | Brand, Model, Year, Mileage, Fuel Type, Engine, Transmission, Accident History |

---

## 🗂️ Project Structure

```
car-price-predictor/
├── app.py                        
├── lasso_regression_model.pkl    
├── Cleaned_Data.csv               
├── requirements.txt  
├── car_price_training.ipynb           
├── .gitignore                    
├── templates/
│   └── index.html                
└── static/
    └── css/
        └── style.css             
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+

### 1. Clone the repository
```bash
git clone https://github.com/Arnaaavvv/car-price-predictor.git
cd car-price-predictor
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```

---

## 🖥️ Features

- **Cascading dropdowns** : Selecting a brand filters the model list. Selecting a model automatically restricts the engine, transmission, and fuel type options to only what's valid for that model
- **Smart validation** : Mileage is validated on the frontend. Non-numeric inputs are caught on the backend with a 400 error response
- **Price floor** : Lasso regression can extrapolate negatively for extreme inputs(year<2000). Thus, the backend uses `max(0, price)` to clamp negatives to $0, then formats the result with commas and 3 decimal places
- **AJAX prediction** : No page reload, results appear inline after clicking Predict
- **Responsive UI** : Centered card layout with a dark gradient background, works on desktop and mobile

---

## 📦 Dependencies

```
flask
pandas
numpy
scikit-learn
joblib
```

---

## ⚠️ Known Limitations

- The model was trained on a **US used car dataset** and thus predictions are in **USD**
- Years before **2000** are excluded from the year dropdown because the dataset has fewer than 20 records per year for 1996–1999, causing the linear model to extrapolate into negative prices for those years
- The Lasso model is **linear** which may underperform on luxury/exotic cars whose pricing follows non-linear patterns
- The `.pkl` was trained with **scikit-learn 1.6.1**, if you're running a newer version, you may see a version warning. Retrain the model to eliminate this
- **No launch year validation** : the form does not prevent a user from selecting a year before the chosen model was launched (e.g. selecting year 2005 for a Tesla Model 3 which only exists from 2017 onwards), which may result in unreliable predictions

---

## 🔧 Retraining the Model

If you want to retrain the model on updated data:

1. Prepare your cleaned CSV with columns: `brand`, `model`, `model_year`, `milage`, `fuel_type`, `engine`, `transmission`, `accident`, `price`
2. Run your training script and save the pipeline:
```python
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso

# Build your pipeline here

joblib.dump(pipeline, 'lasso_regression_model.pkl')
```

3. Replace `lasso_regression_model.pkl` in the project root

---

## 📁 Data

The dataset (`Cleaned_Data.csv`) contains **3,547 used car listings** with the following columns:

| Column | Type | Description |
|---|---|---|
| `brand` | string | Car manufacturer |
| `model` | string | Specific model name |
| `model_year` | int | Year of manufacture |
| `milage` | int | Reading in miles |
| `fuel_type` | int | Encoded: 0=Gasoline, 1=Diesel, 2=Hybrid, 3=Flex Fuel, 4=Plug-In Hybrid |
| `engine` | string | Engine description |
| `transmission` | string | Transmission type |
| `accident` | int | 0 = No accidents, 1 = Has accidents |
| `price` | int | Listed sale price in USD |

---

## 🤝 Contributing

Pull requests are welcome!

---


## 🌐 Data Source

The original dataset before cleaning and preprocessing used in this project was sourced from **Kaggle**:

> [Used Car Price Dataset — Kaggle](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset)

The raw data was cleaned and preprocessed before training, including:
- Encoding `fuel_type` from text labels to integers
- Dropping null values and irrelevant columns
- Selecting the 8 features most relevant to price prediction
- Encoding Accidents : 0 and 1
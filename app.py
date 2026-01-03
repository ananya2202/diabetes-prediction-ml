from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder="templates")

scaler = pickle.load(open("models/scaler.pkl", "rb"))
model_accuracies = pickle.load(open("models/model_accuracies.pkl", "rb"))

best_model_name = max(model_accuracies, key=model_accuracies.get)

model_files = {
    "SVM": "models/svm_model.pkl",
    "KNN": "models/knn_model.pkl",
    "Decision Tree": "models/dt_model.pkl",
    "Random Forest": "models/rf_model.pkl",
    "Naive Bayes": "models/nb_model.pkl",
}

best_model = pickle.load(open(model_files[best_model_name], "rb"))

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

@app.route("/")
def home():
    return render_template("index.html") 

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    try:
        input_values = [data[feature] for feature in FEATURES]
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400

    input_array = np.array(input_values).reshape(1, -1)
    
    # Scale if needed
    if best_model_name in ["SVM", "KNN"]:
        input_array = scaler.transform(input_array)

    prediction = best_model.predict(input_array)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return jsonify({
        "prediction": result,
        "model_used": best_model_name
    })

    prediction = model.predict(input_array)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)

from tensorflow.keras.models import load_model  # pyright: ignore[reportMissingImports]
import joblib
import numpy as np
model_red=joblib.load("red_model.keras")
scaler=joblib.load("scaler_red.pkl")
input_data=np.array([[8.3,0.675,0.26,2.1,0.084,11.0,43.0,0.9976,3.31,0.53,9.2]])
scaled_data=scaler.transform(input_data)
Predicted_quality=model_red.predict(scaled_data)
print("Predicted Quality:",Predicted_quality[0][0])
Accurate_quality=int(Predicted_quality[0][0])
print("Accurate_Quality:",Accurate_quality) 
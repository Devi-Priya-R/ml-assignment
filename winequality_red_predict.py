import joblib
import numpy as np
model=joblib.load("quality.pkl")
scaler=joblib.load("scaler.pkl")
input_data=np.array([[8.3,0.675,0.26,2.1,0.084,11.0,43.0,0.9976,3.31,0.53,9.2]])
scaled_data=scaler.transform(input_data)
Predicted_quality=model.predict(scaled_data)
print("Predicted Quality:",Predicted_quality[0][0])
Rounded_quality=round(Predicted_quality[0][0])
print("Rounded Quality:",Rounded_quality)
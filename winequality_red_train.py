import pandas as pd 
import sklearn.neighbors as ng 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error
from tensorflow.keras.models import Sequential  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Dense  # pyright: ignore[reportMissingImports]
import joblib
import math
#load the dataset
mydata=pd.read_csv("winequality-red.csv")
x=mydata[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
y=mydata["quality"].values #convert to 1d array
#Normalize the data
scaler=StandardScaler()
x=scaler.fit_transform(x)
#splitting the dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#building the model
model_red=Sequential()
model_red.add(Dense(10,activation="relu",input_shape=(11,) ))
model_red.add(Dense(10,activation="relu"))
model_red.add(Dense(10,activation="relu"))
model_red.add(Dense(1))
model_red.compile(optimizer="adam",loss="mse",metrics=["mae"])
model_red.fit(x_train,y_train,epochs=50)
print("Training completed")
#save the model
model_red.save("red_model.keras")
joblib.dump(scaler,"scaler_red.pkl")
#evaluate the model
test_result=model_red.predict(x_test)
print("MSE",mean_squared_error(y_test, test_result))
print("RMSE",math.sqrt(mean_squared_error(y_test, test_result)))
print("MAE",mean_absolute_error(y_test, test_result))
import uvicorn
from fastapi import FastAPI
from SampleData import SampleData
import pickle

pickle_in = open("model.pkl","rb")

model = pickle.load(pickle_in)
app = FastAPI()

@app.get("/")
def root():
    body = {
        "text":"Nigga"
    }

    return body
@app.post("/predict")
def predict(data:SampleData):
    pclass = data.pclass
    sex = ""
    if(data.sex.lower()== "male"):
        sex =1
    else:
        sex = 0
    embarked = ""
    if(data.embarked == "S"):
        embarked = 2
    elif(data.embarked == "C"):
        embarked=0
    else:
        embarked=1

    prediction = model.predict([[pclass,sex,embarked,data.fare,data.age]])
    return {"prediction": int(prediction[0])}

if(__name__ == "__main__"):
    uvicorn.run(app,host = "127.0.0.1",port = 8000)
    print("Running")

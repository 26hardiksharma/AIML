from pydantic import BaseModel

class SampleData(BaseModel):
    pclass:int
    sex:str
    fare:float
    embarked:str
    age:float
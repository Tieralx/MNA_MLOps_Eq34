from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import dvc.api
import joblib  # or another model-loading library as needed
import pickle
from typing import List

with open("./models/lr_model.pkl", "rb") as f:
    #model = pickle.load(f)
    model = joblib.load(f)

# Define input data model for the API
class ca_cervix(BaseModel):
    
    #features: List[int]
    behavior_sexualRisk: int
    behavior_eating: int
    behavior_personalHygine: int
    intention_aggregation: int
    intention_commitment: int
    attitude_consistency: int
    attitude_spontaneity: int
    norm_significantPerson: int
    norm_fulfillment: int
    perception_vulnerability: int
    perception_severity: int
    motivation_strength: int
    motivation_willingness: int
    socialSupport_emotionality: int
    socialSupport_appreciation: int
    socialSupport_instrumental: int
    empowerment_knowledge: int
    empowerment_abilities: int
    empowerment_desires: int

    # Add fields for each feature your model needs

app = FastAPI()

#feature_names = ["behavior_sexualRisk",
#                 "behavior_eating",
#                 "behavior_personalHygine",
#                 "intention_aggregation",
#                 "intention_commitment",
#                 "attitude_consistency",
#                 "attitude_spontaneity",
#                 "norm_significantPerson",
#                 "norm_fulfillment",
#                 "perception_vulnerability",
#                 "perception_severity",
#                 "motivation_strength",
#                 "motivation_willingness",
#                 "socialSupport_emotionality",
#                 "socialSupport_appreciation",
#                 "socialSupport_instrumental",
#                 "empowerment_knowledge",
#                 "empowerment_abilities",
#                 "empowerment_desires"
#                ]


# Load model
#model_path = dvc.api.get_url("./models/lr_model.pkl")
#model = joblib.load(model_path)

@app.post("/predict")
#async def predict(request: ca_cervix):

#    print(request)
async def predict(request: ca_cervix):
    input_data = [request.behavior_sexualRisk,
                   request.behavior_eating,
                   request.behavior_personalHygine,
                   request.intention_aggregation,
                   request.intention_commitment,
                   request.attitude_consistency,
                   request.attitude_spontaneity,
                   request.norm_significantPerson,
                   request.norm_fulfillment,
                   request.perception_vulnerability,
                   request.perception_severity,
                   request.motivation_strength,
                   request.motivation_willingness,
                   request.socialSupport_emotionality,
                   request.socialSupport_appreciation,
                   request.socialSupport_instrumental,
                   request.empowerment_knowledge,
                   request.empowerment_abilities,
                   request.empowerment_desires
                  ]
    #input_data = input_data.reshape(1,1)
    #input_data = request.model_dump
    # Run inference
    
    try:
        prediction = model.predict([input_data])
        
        #return {"prediction": prediction}
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

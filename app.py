from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import dvc.api
import joblib  # or another model-loading library as needed

# Define input data model for the API
class ca_cervix(BaseModel):
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

# Load model
model_path = dvc.api.get_url("models\lr_model.pkl")
model = joblib.load(model_path)

@app.post("/predict")
async def predict(data: ca_cervix):
    # Run inference
    input_data = [[data.behavior_sexualRisk, data.behavior_eating, data.behavior_personalHygine, data.intention_aggregation, data.intention_commitment, data.attitude_consistency, 
        data.attitude_spontaneity, data.norm_significantPerson, data.norm_fulfillment, data.perception_vulnerability, data.perception_severity, data.motivation_strength, 
        data.motivation_willingness, data.socialSupport_emotionality, data.socialSupport_appreciation, data.socialSupport_instrumental, data.empowerment_knowledge, 
        data.empowerment_abilities, data.empowerment_desires
        ]]  # Adapt as per your model's input requirements
    try:
        prediction = model.predict(input_data)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

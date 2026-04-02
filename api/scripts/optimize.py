# SCRIPT : optimize.py
import dspy
from dspy.teleprompt import BootstrapFewShot
from scripts.data_training_dspy import trainset

import time
time.sleep(15)

# 1. Config
vllm = dspy.LM(
    model="openai/medical_lora", 
    api_base="http://vllm:8000/v1", 
    api_key="EMPTY"
)
dspy.settings.configure(lm=vllm)

# 2. La Signature (Le contrat)
class TriageSignature(dspy.Signature):
    """Assistant de triage. Si assez d'infos -> ### ANALYSE + JSON. Sinon -> ### ASSISTANT + Question."""
    symptomes = dspy.InputField()
    reponse = dspy.OutputField()

# 3. Le Module (Le moteur)
class TriageModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(TriageSignature)
    
    def forward(self, symptomes):
        return self.predictor(symptomes=symptomes)

# 4. Compilation
optimizer = BootstrapFewShot(metric=None, max_bootstrapped_demos=2)
compiled_program = optimizer.compile(TriageModule(), trainset=trainset)
compiled_program.save("optimized_triage.json")



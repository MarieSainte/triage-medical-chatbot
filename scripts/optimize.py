# SCRIPT : optimize.py
import dspy
from dspy.teleprompt import BootstrapFewShot
from scripts.data_training_dspy import trainset
# 1. Config
vllm = dspy.HFClientVLLM(model="medical_lora", url="http://localhost:8000")
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

# 4. Compilation (On crée la recette avec tes exemples 'trainset')
optimizer = BootstrapFewShot(metric=None)
compiled_program = optimizer.compile(TriageModule(), trainset=trainset)

# 5. Sauvegarde (C'est ce fichier qu'on veut !)
compiled_program.save("optimized_triage.json")



import os
from src.model import RCClassifier

# Ensure HF token is defined as env var
os.environ["HF_TOKEN"]


model = RCClassifier.load_from_checkpoint("logs/exp-1/version_0/checkpoints/epoch44-step225-loss6.320-f13.588.ckpt")
model.push_to_hub("jatwell/rc-classifier")

# test the download works
# model = RCClassifier.from_pretrained("jatwell/rc-classifier")

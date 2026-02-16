
---

### **`weights/Facilitator/README.md`**

```markdown
# Facilitator Pre-trained Weights

This folder will contain the pre-trained weights for the **Facilitator** model. The Facilitator model is part of the BioM3 pipeline and serves as a key component for further alignment or generation tasks.

---

## **Downloading Pre-trained Weights**

The Google Drive link for downloading the Facilitator pre-trained weights will be added here soon.


```bash
pip install gdown # assuming gdown package is not already installed
gdown --id 1_YWwILXDkx9MSoSA1kfS-y0jk3Vy4HJE -O BioM3_Facilitator_epoch20.bin
```

---

## **Usage**

Once available, the pre-trained weights can be loaded as follows:

```python
import json
import torch
from argparse import Namespace
import Stage1_source.model as mod
  
# Step 1: Load JSON Configuration
def load_json_config(json_path):
    """
    Load a JSON configuration file and return it as a dictionary.
    """
    with open(json_path, "r") as f:
        config = json.load(f)
    return config

# Step 2: Convert JSON Dictionary to Namespace
def convert_to_namespace(config_dict):
    """
    Recursively convert a dictionary to an argparse Namespace.
    """
    for key, value in config_dict.items():
        if isinstance(value, dict):
            config_dict[key] = convert_to_namespace(value)
    return Namespace(**config_dict)

if __name__ == '__main__':
    # Path to configuration and weights
    config_path = "stage2_config.json"
    model_weights_path = "weights/Facilitator/BioM3_Facilitator_epoch20.bin"

    # Load Configuration
    print("Loading configuration...")
    config_dict = load_json_config(config_path)
    config_args = convert_to_namespace(config_dict)

    # Load Model
    print("Loading pre-trained model weights...")
    model = mod.Facilitator(
        in_dim=config_args.emb_dim,
        hid_dim=config_args.hid_dim,
        out_dim=config_args.emb_dim,
        dropout=config_args.dropout
    ) # Initialize the model with arguments
    model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
    model.eval()
    print("Model loaded successfully with weights!")

```

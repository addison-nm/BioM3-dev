
---

### **`weights/PenCL/README.md`**

```markdown
# PenCL Pre-trained Weights

This folder contains the pre-trained weights for the **PenCL** model (Stage 1 of BioM3). The PenCL model aligns protein sequences and text descriptions to compute joint latent embeddings.

---

## **Downloading Pre-trained Weights**

To download the **PenCL epoch 20 pre-trained weights** as a `.bin` file from Google Drive, use the following command:

```bash
pip install gdown
gdown --id 1Lup7Xqwa1NjJpoM2uvvBAdghoM-fecEj -O BioM3_PenCL_epoch20.bin
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
    config_path = "stage1_config.json"
    model_weights_path = "weights/PenCL/BioM3_PenCL_epoch20.bin"

    # Load Configuration
    print("Loading configuration...")
    config_dict = load_json_config(config_path)
    config_args = convert_to_namespace(config_dict)

    # Load Model
    print("Loading pre-trained model weights...")
    model = mod.pfam_PEN_CL(args=config_args)  # Initialize the model with arguments
    model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
    model.eval()
    print("Model loaded successfully with weights!")
```

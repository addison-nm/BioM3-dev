
---

### **`weights/ProteoScribe/README.md`**

```markdown

# ProteoScribe Pre-trained Weights
This folder contains the pre-trained weights for the **ProteoScribe** model (Stage 3 of BioM3). The ProteoScribe model generates protein sequences from conditioned latent embeddings.

---
## **Downloading Pre-trained Weights**
To download the **ProteoScribe epoch 20 pre-trained weights** as a `.bin` file from Google Drive, use the following command:
```bash
pip install gdown
gdown --id 1c3CwvbOP_kp3FpLL1wPrjO6qtY-XiT26 -O BioM3_ProteoScribe_pfam_epoch20_v1.bin
```

---
## **Usage**
Once available, the pre-trained weights can be loaded as follows:
```python
import json
import torch
import torch.nn as nn
from argparse import Namespace
import Stage3_source.cond_diff_transformer_layer as Stage3_mod

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

# Step 3: Model Loading Function
def prepare_model(model_path, config_args) -> nn.Module:
    """
    Initialize and load the ProteoScribe model with pre-trained weights.
    """
    # Initialize the model graph
    model = Stage3_mod.get_model(
        args=config_args,
        data_shape=(config_args.image_size, config_args.image_size),
        num_classes=config_args.num_classes
    )

    # Load pre-trained weights
    model.load_state_dict(torch.load(model_path, map_location=config_args.device))
    model.eval()
    
    return model

if __name__ == '__main__':
    # Path to configuration and weights
    config_path = "stage3_config.json"
    model_weights_path = "weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin"

    # Load Configuration
    print("Loading configuration...")
    config_dict = load_json_config(config_path)
    config_args = convert_to_namespace(config_dict)
    
    # Set device if not specified in config
    if not hasattr(config_args, 'device'):
        config_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Model
    print("Loading pre-trained model weights...")
    model = prepare_model(model_weights_path, config_args)
    print(f"Model loaded successfully with weights! (Device: {config_args.device})")
```

---
## **Model Structure**
The ProteoScribe model is structured as a conditional diffusion transformer that generates protein sequences based on facilitated embeddings. The model consists of:

1. A transformer-based architecture for sequence generation
2. Conditional diffusion layers for embedding processing
3. Output layers for amino acid sequence prediction

---
## **Configuration Requirements**
The `stage3_config.json` file should contain the following key parameters:

```json
{
    "image_size": [required_size],
    "num_classes": [num_amino_acids],
    "device": "cuda",  // or "cpu"
    // Additional model-specific parameters
}
```

---
## **Dependencies**
Ensure you have the following dependencies installed:
- PyTorch (latest stable version)
- Stage3_source module (included in the BioM3 repository)

---
## **Important Notes**
1. The model expects facilitated embeddings (z_c) as input, typically generated from Stage 2 (Facilitator)
2. Model weights are optimized for protein sequence generation tasks
3. Use CUDA-enabled GPU for optimal performance (if available)
4. Default configuration is tuned for the Pfam database

---
## **Troubleshooting**
Common issues and solutions:

1. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Use CPU if GPU memory is insufficient

2. **Module Import Errors**
   - Ensure Stage3_source is in Python path
   - Check all dependencies are installed

3. **Weight Loading Issues**
   - Verify the downloaded weights file is complete
   - Check model configuration matches pre-trained architecture

For additional support or issues:
- Open an issue in the BioM3 repository
- Check the documentation for updates

---
## **Citation**
If you use these weights in your research, please cite:
```bibtex
Natural Language Prompts Guide the Design of Novel Functional Protein Sequences
bioRxiv 2024.11.11.622734
doi: https://doi.org/10.1101/2024.11.11.622734
```

---
Repository maintained by the BioM3 Team

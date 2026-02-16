# Weights Directory

This folder contains the pre-trained weights for the **BioM3** project models. The weights are stored as `.bin` files for different components of the BioM3 pipeline:

0. **LLMs**: Pre-trained weights for compiling PenCL model. 
1. **PenCL**: Pre-trained weights for the PenCL model (Stage 1).
2. **Facilitator**: Pre-trained weights for the Facilitator model (Stage 2).
3. **ProteoScribe**: Pre-trained weights for the ProteoScribe model (Stage 3).

---

## **Purpose**

The weights provided here enable users to quickly load and run inference with the pre-trained models for text-protein sequence alignment, functional annotation, and other tasks.

Each subfolder includes:
- Instructions for downloading the desired `.bin` files.
- Information on integrating the weights into your workflows.

---

### **Prerequisites**

To download pre-trained weights, you must install `gdown`:

```bash
pip install gdown


# **Tiny-Storyteller: Optimized Fine-Tuning of DistilGPT2 with Quantization Techniques**  

## **Overview**  
This project fine-tunes **DistilGPT2** on the **TinyStories dataset** to create a **lightweight, memory-efficient language model** for **short-form story generation**.  
The focus is on **efficient quantization** using:  
✅ **BitsAndBytes (4-bit NF4 quantization)**  
✅ **HQQ (1-bit quantization for extreme compression)**  
✅ **QLoRA (PEFT-based Low-Rank Adaptation for training efficiency)**  

By leveraging **quantization and parameter-efficient fine-tuning**, we achieve **significant memory savings** while maintaining **story generation quality**.

---

## **Key Features**  
- **Dataset**: Uses **TinyStories** (2.1M training samples)  
- **Base Model**: **DistilGPT2**, a lightweight version of GPT-2  
- **Quantization Methods**:  
  - **BitsAndBytes (bnb)**: 4-bit **NF4 quantization** for efficient inference  
  - **HQQ (1-bit quantization)**: Extreme memory compression for edge deployment  
  - **QLoRA (LoRA + QLoRA adapters)**: Enables **fine-tuning on consumer GPUs**  
- **Training Setup**:  
  - **Per-device batch size: 4**  
  - **FP16 training** for optimized computation  
  - **Gradient accumulation** for memory efficiency  
- **Deployment**: Model is pushed to **Hugging Face Hub**  

---

## **Quantization & Efficiency Improvements**  

| **Method**     | **Memory Footprint (MB)** | **Fine-Tuning Feasibility** | **Inference Speed** |
|---------------|--------------------------|----------------------------|---------------------|
| **FP32 Baseline** | ~350MB | Expensive | Slow |
| **BitsAndBytes (4-bit)** | ~90MB | Moderate | Faster |
| **HQQ (1-bit)** | ~40MB | Challenging | Very Fast |
| **QLoRA (4-bit)** | ~110MB | High | Faster |

- **HQQ achieves up to 10x compression** but may require trade-offs in precision.  
- **QLoRA provides a balance** between compression and fine-tuning capability.  
- **BitsAndBytes (bnb) NF4** enables near **full-precision performance** at a fraction of memory usage.  

---

## **Results & Findings**  

### **Memory Savings**  
- **QLoRA (4-bit)** enabled training on a **single consumer GPU (12GB VRAM)**  
- **BitsAndBytes 4-bit NF4** provided near **FP32 performance** while reducing **memory usage by ~75%**  
- **HQQ (1-bit)** reduced model footprint **to <50MB**, making it viable for **edge deployment**  

### **Performance Trade-Offs**  
- **BitsAndBytes (4-bit)**: Maintained >95% of FP32 accuracy  
- **QLoRA**: Enabled fine-tuning with minimal memory overhead  
- **HQQ (1-bit)**: **Drastic memory savings but loss of generation fluency**  

---

## **Future Work**  
🔹 **Hybrid Quantization**: Combining **HQQ for early layers** & **QLoRA for late layers**  
🔹 **Knowledge Distillation**: Compressing **DistilGPT2 further** without loss in generation quality  
🔹 **Exploring Grouped Quantization**: Testing different **group sizes** in **HQQ and QLoRA**  

---

## **How to Use the Model?**  
You can **fine-tune and deploy** the model from **Hugging Face Hub**:  
🔗 [Tiny-Storyteller Model](https://huggingface.co/srinivasan-sridhar28/Tiny-Storyteller)  

```python
from transformers import pipeline
story_generator = pipeline("text-generation", model="srinivasan-sridhar28/Tiny-Storyteller")
story_generator("Once upon a time", max_length=50)
```




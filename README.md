# Improving-Robustness-of-Deepfake-Detectors-through-Gradient-Regularization

# OVERVIEW

Recent generative models are capable of creating highly realistic deepfakes, posing significant threats to privacy, trust, and digital security. This project addresses the challenge of building robust deepfake detectors by applying gradient regularization to improve both generalization and resistance to adversarial attacks. Our approach builds on EfficientNet-B0 and introduces a perturbation-based training method that encourages the model to learn stable representations.

# DATASET

We use the DFFD (DeepFake Face Dataset) consisting of:
-Real faces: from FFHQ;
-Face faces: from various methods
    -FaceApp
    -StyleGAN
    -StarGAN
    -PG-GAN
The dataset is split into:
-Training: real+ fake images (balanced via weights)
-Validation
-Test

# KEY FEATURES

- EfficientNet-B0: baseline with pretrained weights
- Gradient Regularization based on shallow feature perturbation
- Support for 3 adversarial atttacks :
  -FGSM
  -PGD
  -BIM
-Evaluation metrics:
  -Accuracy
  -Macro F1 score
  -IINC
  -AUC
-CAM Visualizations
-Plots for  performance comparison

# ARCHITECTURE & MODELS

Two models are trained:
-baseline
-Regularized


## baseline model

The baseline model uses a pretrained EfficientNet-B0 from ImageNet with the final classification head replaced by  Linear layer for binary classification. It is trained using a class-weighted cross-entropy loss to address the class imbalance between real and fake samples

## graident-regularized Model


To enhance robustness, a regularized variant is introduced:
- The model is **split** into two parts:
  - `features_early`: initial convolutional blocks  
  - `features_late`: deeper layers and classifier  
- During training, **two forward passes** are computed:
  - `forward_clean(x)`: standard inference  
  - `forward_perturbed(x)`: perturbs shallow features using random noise on their **mean** and **standard deviation**, simulating feature-level domain shift
 
The total loss function is computed as:
L_total = (1 - alpha) * L_clean + alpha * L_perturbed

Where:
- `alpha` balances the contribution of regularization (e.g., 0.5)  
- `epsilon` controls the strength of the perturbation (e.g., 0.05)  
- Both components use standard **cross-entropy loss**

##  Adversarial Evaluation

Robustness is tested using three adversarial attacks on both the baseline and the regularized model:

- **FGSM**  
- **PGD**  
- **BIM**

Performance is measured across different epsilon values (perturbation strength), and comparisons are made in terms of accuracy, F1 score, and IINC to assess model stability under attack.

## Visual Analysis

- **Grad-CAM** is used to visualize class activation maps under clean and adversarial settings  
- **IINC** metric quantifies the change in CAM between clean and perturbed inputs  
- Helps assess the model’s interpretability and resilience

---

## Results Summary

- Regularized models show **higher robustness** across FGSM, PGD, and BIM attacks  
- **IINC** is consistently lower in the regularized model, showing better feature invariance  
- Accuracy and F1 remain more stable as epsilon increases





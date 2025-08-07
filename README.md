# Pulmonary-Fibrosis-Early-Detection-
A CNN-based pipeline for early detection of pulmonary fibrosis from lung sound recordings. Converts audio to Mel spectrograms, performs binary classification (Fibrosis vs Non-Fibrosis), and includes scripts for preprocessing, training, and testing.     

# Label Mapping for Binary Classification

The following table maps original dataset folders to binary class labels used for training the fibrosis detection model:

| **Original Folder**       | **New Label**       | **Notes**                                   |
|---------------------------|---------------------|---------------------------------------------|
| `Fibrosis (15)`           | `1 (Fibrosis)`      | Confirmed fibrosis samples                  |
| `Pseudo Fibrosis (75)`    | `1 (Fibrosis)`      | Derived from crepitations, assumed fibrosis |
| `Crepitations (229)`      | `1 (Fibrosis)`      | Or partial fibrosis                         |
| `Asthma (99)`             | `0 (Non-Fibrosis)`  | Excluded from fibrosis category             |
| `Crackles (15)`           | `0 (Non-Fibrosis)`  | Or discarded                                |
| `Normal_Sounds (21)`      | `0 (Non-Fibrosis)`  | Confirmed healthy lung sounds               |

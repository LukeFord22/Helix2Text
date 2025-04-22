# Helix2Text

## Overview
Helix2Text is a project designed to integrate DNA sequence data with natural language descriptions. It leverages advanced machine learning models, such as METAGENE-1-BnB-4Bit and DistilGPT2, to process DNA sequences and generate meaningful text annotations or vice versa. This project is particularly useful for bioinformatics applications, such as annotating genetic data, generating textual descriptions of DNA features, and facilitating the understanding of genomic information.

## Features
- **DNA-to-Text Generation**: Converts DNA sequences into descriptive text using transformer-based models.
- **Text-to-DNA Generation**: Generates DNA sequences based on textual descriptions.
- **Integration of Multiple Models**: Combines models like METAGENE-1-BnB-4Bit and DistilGPT2 for efficient processing.
- **Support for Quantized Models**: Utilizes 4-bit quantization for efficient computation on large-scale datasets.
- **Customizable Training and Inference**: Provides flexibility for fine-tuning and inference on custom datasets.
- **Toxicity Prediction**: Ability to analyze DNA sequences for potential toxic properties.

## Project Structure
- **Helix2Text/**: Core directory for the Helix2Text project, including model definitions and training scripts.
  - `METAGENE-1model.py`: Main implementation of the DNA-text fusion model.
  - `requirements.txt`: Dependencies specific to Helix2Text.

## Key Files
- **METAGENE-1model.py**: Main script for integrating DNA and text models.


## Datasets
The project utilizes several specialized datasets:


## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r Helix2Text/requirements.txt
   ```
3. Download pre-trained models:
   - METAGENE-1-BnB-4Bit can be downloaded from Hugging Face
   - DistilGPT2 will be automatically downloaded through the Transformers library

## Usage
### Training
To train the Helix2Text model, run the following command:
```bash
python Helix2Text/METAmodel.py
```

## Dependencies
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- BitsAndBytes (for quantized models)
- Pandas
- Scikit-learn
- Datasets (Hugging Face)
- Accelerate (for multi-GPU support)
- TQDM (for progress tracking)

## Model Architecture
The Helix2Text model uses a fusion architecture that combines:
1. A DNA sequence model (METAGENE-1-BnB-4Bit) that processes genomic data
2. A text language model (DistilGPT2) that handles natural language
3. Projection layers to facilitate communication between the two models

This architecture enables bidirectional translation between DNA sequences and human-readable text descriptions.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any bugs or feature requests.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
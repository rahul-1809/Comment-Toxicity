# Toxic Comment Detection Using Deep Learning

## Overview
This project focuses on building a deep learning model to detect and classify toxic comments into multiple categories such as `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. By leveraging TensorFlow, Python, and Deep-learning techniques, this project provides a robust pipeline for toxicity detection and integrates a user-friendly interface for real-time predictions.

## Features
- Preprocesses text data with a custom `TextVectorization` pipeline.
- Implements a Bidirectional LSTM model to classify comments into multiple categories.
- Splits data into training, validation, and test sets to ensure robust evaluation.
- Interactive Gradio-based web interface for real-time toxicity scoring.
- Saves and reloads trained models for scalability and deployment.

## Dataset
The dataset used for this project is a multi-label classification dataset where each comment is labeled with one or more of the following categories:
- **toxic**: General toxic behavior.
- **severe_toxic**: Highly aggressive or harmful content.
- **obscene**: Usage of vulgar or explicit language.
- **threat**: Threats or intimidation.
- **insult**: Personal attacks or demeaning comments.
- **identity_hate**: Hate speech targeting specific identities (e.g., race, religion, gender).

### Data Preprocessing
- Text data is processed using TensorFlow's `TextVectorization` to handle up to 200,000 unique words.
- The maximum sequence length is capped at 1800 tokens to ensure efficient model training.

## Model Architecture
The model is built using TensorFlow's Sequential API and includes the following layers:
- **Embedding Layer**: Converts words into dense vector representations.
- **Bidirectional LSTM Layer**: Captures long-term dependencies in the text.
- **Dense Layers**: Adds non-linearity and captures complex interactions in the data.
- **Output Layer**: Produces predictions for each toxicity class using sigmoid activation.

## Deployment
A Gradio-based interface is implemented to provide real-time predictions. The trained model is saved in the `.h5` format for reuse and easy deployment.

### Clone the repository:
   ```bash
   git clone https://github.com/rahul-1809/Comment-Toxicity/
   ```

## Dependencies
- Python 3.8+
- TensorFlow
- NumPy
- Pandas
- Gradio
- Matplotlib

## Acknowledgments
- The dataset used in this project is based on the Jigsaw Toxic Comment Classification Challenge.
- Thanks to Nicholas Renotte and the open-source contributors of TensorFlow and Gradio for their tools and resources.

## License
This project is licensed under the MIT License. See the LICENSE file for details.


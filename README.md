# **Stress Prediction Project**

This project implements a machine learning pipeline to predict stress levels from physiological and behavioral data. The system uses RNN models optimized with Keras Tuner to find the best hyperparameters for stress classification.

## **Project Overview**

This document provides an overview of the project, including its structure, setup instructions, usage, and details about the machine learning pipeline.

## **Project Structure**

The project is organized as follows:

```
.  
├── assets/                      # Contains trained models and configuration files  
│   ├── category_mapping.json    # Mapping for categorical variables  
│   └── scaler.pkl               # Saved StandardScaler for data normalization  
├── Dataset/                     # Data storage  
│   ├── combined_data.csv        # Original dataset  
│   └── combined_data_preprocessed.csv  # Preprocessed dataset  
├── function/                    # Core functionality modules  
│   ├── hypermodel.py            # Hypermodel definition for Keras Tuner  
│   ├── preprocess.py            # Data preprocessing utilities  
│   └── test_gpu.py              # GPU availability check  
├── my_tuner_results/            # Keras Tuner results storage  
├── result/                      # Model evaluation results  
├── main.py                      # Main execution script  
├── requirements.txt             # Project dependencies  
└── tuner_results_summary.txt    # Summary of hyperparameter tuning results
```

## **Setup and Installation**

Follow these steps to set up the project environment:

1. Clone the repository:  
  ```bash
  git clone <your-repository-url>  
  cd <repository-name>
  ```

2. Create a Python virtual environment:  
  It's highly recommended to use a virtual environment to manage project dependencies and avoid conflicts.  
  * **Using venv (Python 3 built-in):**  
    ```bash
    python3 -m venv venv

    # Activate the environment:  
    # On macOS and Linux:  
    source venv/bin/activate

    # On Windows:  
    .\venv\Scripts\activate
    ```

  * **Using conda (if you have Anaconda/Miniconda installed):**  
    ```bash
    # Replace stress_env with your preferred environment name
    conda create --name stress_env python=3.8  
    conda activate stress_env
    ```

3. Install the required dependencies:  
  ```bash
  pip install -r requirements.txt
  ```

## **Data Preparation**

The project expects a CSV dataset (`combined_data.csv` located in the `Dataset/` directory) with the following features:

* **Physiological signals:** Examples include EDA_Phasic, SCR_Amplitude, NumPeaks, HRV (Heart Rate Variability) metrics, etc.  
* **Personal attributes:** Examples include gender, BMI, sleep duration/quality, etc.  
* **Target variable:** stress level (this will be the variable the model aims to predict).

The raw data will be preprocessed, and the output will be saved as `combined_data_preprocessed.csv`.

## **Usage**

To run the complete machine learning pipeline, execute the main script:

```bash
python main.py
```

This script will perform the following steps:

1. Check for GPU availability to leverage hardware acceleration if possible.  
2. Load the dataset from `Dataset/combined_data.csv`.  
3. Preprocess the data, including feature scaling and encoding.  
4. Create sequences suitable for RNN model input.  
5. Rebalance the dataset using SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.  
6. Split the data into training and testing sets.  
7. Perform hyperparameter tuning using Keras Tuner with Bayesian Optimization.  
8. Save the tuning results and the best model.

## **Preprocessing Pipeline**

The data preprocessing pipeline involves several key steps:

* **Column selection:** Relevant features are selected for model training.  
* **Label encoding:** Categorical variables (e.g., 'gender') are converted into numerical representations.  
* **Feature scaling:** Numerical features are standardized using StandardScaler to have zero mean and unit variance. This helps in improving the performance of many machine learning algorithms.  
* **Sequence creation:** Data is transformed into sequences, which is a necessary format for time-series analysis with RNNs.  
* **Class balancing (SMOTE):** Addresses potential imbalances in the distribution of stress levels in the dataset by generating synthetic samples for minority classes.

## **Hyperparameter Tuning**

Keras Tuner with Bayesian Optimization is employed to find the optimal set of hyperparameters for the RNN model. The tuning process searches for the best:

* **Number of RNN units:** The dimensionality of the output space for RNN layers (e.g., LSTM, GRU).  
* **Dropout rate:** The fraction of input units to drop during training to prevent overfitting.  
* **Number of layers:** The depth of the RNN model.  
* **Optimizer choice:** The optimization algorithm to use (e.g., Adam, RMSprop).  
* **Learning rate:** The step size for the optimizer.

## **Results**

The outcomes of the hyperparameter tuning and model evaluation are stored as follows:

* **Keras Tuner Results:** Detailed logs and checkpoints from the tuning process are saved in the `my_tuner_results/` directory.  
* **Tuning Summary:** A summary of the best hyperparameters and their corresponding performance metrics is written to `tuner_results_summary.txt`.  
* **Model Evaluation:** Evaluation results, potentially including confusion matrices, classification reports, and plots, are saved in the `result/` directory.

## **Extending the Project**

To adapt or extend this project for different datasets or requirements:

1. **Modify Column List:** Update the list of columns to be used from your dataset in `main.py` (and potentially in `function/preprocess.py`).  
2. **Adjust Preprocessing:** If your new dataset has different characteristics (e.g., different types of categorical features, other missing value patterns), modify the preprocessing steps in `function/preprocess.py` accordingly.  
3. **Update Hypermodel:** If you want to try different types of RNN layers, different ranges for hyperparameters, or a different model architecture, update the hypermodel definition in `function/hypermodel.py`.

## **Troubleshooting**

* **"epochs does not exist" error in tuner_results_summary.txt:** This usually indicates that the epochs parameter was not correctly defined or accessed within your Keras Tuner HyperModel class or the tuning script. Ensure that the fit method call within the tuner's search is correctly passing or accessing the number of epochs.  
* **Dependency Issues:** Ensure all packages in `requirements.txt` are installed correctly in your active virtual environment. If you encounter module not found errors, try reinstalling the specific package or the entire `requirements.txt`.  
* **GPU Issues:** If `test_gpu.py` indicates no GPU is found or you encounter CUDA errors, ensure your GPU drivers and CUDA toolkit are correctly installed and compatible with your TensorFlow version.

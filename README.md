# Marine Microplastics Analysis

This repository contains a project focused on analyzing marine microplastics using machine learning models. The project leverages data in the WGS84 coordinate system and includes tools for preprocessing, training, and deploying a predictive model.

## Project Structure

- **Marine_Microplastics_WGS84_-4298210065197307901.csv**: The dataset used for training and testing the model.
- **marine_model.pkl**: The trained machine learning model.
- **Marine_streamlit_simple.py**: A Streamlit application for visualizing and interacting with the model.
- **model_columns.pkl**: A pickle file containing the column names used in the model.
- **model_info.pkl**: Metadata about the trained model.
- **model_trainer.py**: A Python script for training the machine learning model.
- **scaler.pkl**: A pickle file containing the scaler used for data normalization.

## Features

- **Data Preprocessing**: Includes scaling and cleaning of the dataset.
- **Model Training**: A script to train a machine learning model on the provided dataset.
- **Model Deployment**: A Streamlit app for deploying and visualizing the model's predictions.

## Requirements

To run this project, you need the following Python packages:

- pandas
- numpy
- scikit-learn
- streamlit

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

1. **Train the Model**:
   Run the `model_trainer.py` script to train the model. The trained model will be saved as `marine_model.pkl`.

   ```bash
   python model_trainer.py
   ```

2. **Run the Streamlit App**:
   Launch the Streamlit app to visualize and interact with the model.

   ```bash
   streamlit run Marine_streamlit_simple.py
   ```

## Dataset

The dataset `Marine_Microplastics_WGS84_-4298210065197307901.csv` contains geospatial data in the WGS84 coordinate system. Ensure the dataset is in the correct format before running the scripts.

## Model

The machine learning model is trained to predict microplastic concentrations based on the provided dataset. The model's metadata and column information are stored in `model_info.pkl` and `model_columns.pkl`, respectively.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## Contact

For any questions or issues, please open an issue in this repository.

# Mushroom Classification with Logistic Regression

## Overview

This repository contains the implementation of a mushroom classification project using logistic regression. The goal of the project is to classify mushrooms as either edible or poisonous based on a dataset of mushroom characteristics. The project is implemented in Python and follows a structured approach, including data extraction, transformation, modeling, and evaluation.

## Project Structure

The project is organized into the following directories and files:

-   `dataset/`: Contains the raw mushroom dataset.
-   `etl/`: Contains scripts for data extraction, transformation, and loading.
-   `model/`: Contains the implementation of the logistic regression model.
-   `Report.pdf`: Contains the LaTeX report of the project.

## Setup and Installation

To get started with this project, follow these steps:

1.  **Clone the repository:**

    `git clone https://github.com/dembA7/Binary-Classification.git`

2.  **Install the required packages:**

    You can install the required Python packages using pip. Make sure you have `pandas`, `numpy`, `matplotlib`, and `seaborn` installed.

    `pip install pandas numpy matplotlib seaborn`

3.  **Prepare the data:**

    Run the ETL script to prepare the data:

    `python etl/etl.py`

4.  **Train the model:**

    Run the main script to train and evaluate the logistic regression model:

    `python model/main.py`

## Project Details

### Data

The dataset used for this project is the Mushroom Dataset from the UCI Machine Learning Repository. It contains various attributes related to mushrooms, such as color, odor, and habitat, and is used to predict whether a mushroom is edible or poisonous.

### Model

The logistic regression model, contained in the file `model/model.py`, is implemented to classify mushrooms based on the features provided. The implementation includes:

-   **Sigmoid Function**: For computing probabilities.
-   **Cost Function**: Cross-entropy loss function.
-   **Gradient Descent**: For optimizing the model parameters.
-   **Evaluation**: Accuracy and confusion matrix.

### Visualizations

The `model/graphs.py` file contains scripts for generating various visualizations, including:

-   **Confusion Matrix**: Visualizing the confusion matrix.
-   **Probability Distributions**: Visualizing the confidence of predictions.
-   **Cost Evolution**: Showing the progression of the cost function during training.

### Usage

After setting up the environment and running the scripts, you can find the following outputs:

-   **Model Metrics**: Accuracy, cost and other performance metrics.
-   **Visualizations**: Plots for probability distributions, cost evolution and matrix confusions.

### License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.

### Acknowledgments

-   The Mushroom Dataset from the UCI Machine Learning Repository.
-   Various online resources and documentation used for implementing the model and visualizations.

# Customers-Clustering-With-K-Means

Customers-Clustering-With-K-Means is a project that utilizes the K-Means clustering algorithm to group customers based on their purchasing behavior and other relevant features. This can help businesses understand their customer segments better and tailor their marketing strategies accordingly.

## Introduction

This project aims to perform customer segmentation using the K-Means clustering algorithm. By analyzing customer data, such as purchase history, demographic information, and other relevant metrics, we can identify distinct groups of customers with similar characteristics.

## Features

- Data preprocessing and cleaning
- Implementation of the K-Means clustering algorithm
- Visualization of clustered data
- Evaluation of clustering performance
- Support for custom datasets

## Data Description

The dataset contains customer information with the following columns:

- **CustomerID**: Unique identifier for each customer
- **Gender**: Gender of the customer
- **Age**: Age of the customer
- **Annual Income (k$)**: Annual income of the customer in thousand dollars
- **Spending Score (1-100)**: Score assigned to the customer based on their spending behavior

### Sample Data

| CustomerID | Gender | Age | Annual Income (k$) | Spending Score (1-100) |
|------------|--------|-----|--------------------|------------------------|
| 1          | Male   | 19  | 15                 | 39                     |
| 2          | Male   | 21  | 15                 | 81                     |
| 3          | Female | 20  | 16                 | 6                      |
| 4          | Female | 23  | 16                 | 77                     |
| 5          | Female | 31  | 17                 | 40                     |

### Data Source

The data used in this project is sourced from the Kaggle dataset [Customer Segmentation Tutorial in Python](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python). Ensure your dataset is in a similar format for compatibility.

## Installation

To get started with this project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Customers-Clustering-With-K-Means.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
## Acknowledgements

- Special thanks to scikit-learn for their excellent machine learning library.
- Data sourced from the Kaggle dataset Customer Segmentation Tutorial in Python.
- Inspiration from various online tutorials and courses on K-Means clustering.

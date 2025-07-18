# Predicting Airbnb Rental Prices Using Machine Learning

This project focuses on predicting Airbnb rental prices in New York City using various machine learning techniques. The primary goal is to identify the most accurate model for price prediction by comparing the performance of Ridge Regression, Support Vector Regression (SVR), Multi-layer Perceptron (MLP), and XGBoost.

## 📖 Project Description

Airbnb has become a major player in the hospitality industry. For both hosts and guests, understanding the factors that influence rental prices is crucial. This project tackles the challenge of accurately predicting Airbnb listing prices by leveraging machine learning. We use a dataset of Airbnb listings from New York City and apply several advanced regression models to see which one performs best. The project involves extensive data cleaning, feature engineering, and sentiment analysis of customer reviews to enrich the dataset and improve model accuracy.

## 📊 Dataset

The dataset used in this study is from public Airbnb data for New York City, specifically from August to November 2024. It consists of two main files: `listings.csv` and `reviews.csv`.

### Data Preprocessing and Feature Engineering
To prepare the data for modeling, the following steps were taken:
- **Data Cleaning:** Unnecessary columns were dropped, and missing values were handled.
- **Feature Engineering:** New features were created, such as the host's tenure. Categorical features were one-hot encoded, and the `amenities` column was processed to create binary features. The price was log-transformed to handle its skewed distribution.
- **Sentiment Analysis:** The `TextBlob` library was used to perform sentiment analysis on the text of the reviews. The average sentiment score for each listing was calculated and added as a new feature.
- **Feature Selection:** Lasso regression was used to select the most relevant features for the models.

## 🤖 Methodology

We implemented and evaluated four different machine learning models:

1.  **Ridge Regression:** A linear model that includes L2 regularization to prevent overfitting. It served as our baseline model.
2.  **Support Vector Regression (SVR):** A powerful regression algorithm that works well for non-linear relationships. We used the Radial Basis Function (RBF) kernel.
3.  **Multi-layer Perceptron (MLP):** A feedforward artificial neural network capable of learning complex, non-linear patterns in the data.
4.  **XGBoost:** A highly efficient and scalable implementation of gradient boosting that is known for its high performance in predictive modeling tasks.

Hyperparameter tuning was performed for all models to optimize their performance.

## 📈 Results

The performance of each model was evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared ($R^2$) on both the training and test sets.

| Model              | Test MAE | Test MSE | Test R² |
| ------------------ | -------- | -------- | ------- |
| Ridge Regression   | 0.3161   | 0.1970   | 0.6968  |
| SVR                | 0.1696   | 0.0894   | 0.8623  |
| MLP                | 0.1304   | 0.0519   | 0.9200  |
| **XGBoost** | **0.1052** | **0.0385** | **0.9407** |

The **XGBoost** model outperformed all other models, demonstrating its superior predictive power for this task.

## 🔮 Future Work

Future improvements for this project could include:
-   **Additional Features:** Incorporating more features like local events, public transport proximity, and more detailed seasonal trends.
-   **Ensemble Methods:** Combining the predictions of multiple models to potentially improve accuracy.
-   **Advanced Hyperparameter Tuning:** Using more sophisticated techniques like Bayesian optimization for all models.
-   **Overfitting Mitigation:** Applying techniques like dropout and more advanced regularization to the MLP and SVR models.

## 👥 Contributors
-   Tran Huynh Anh Phuc
-   Nguyen Tran Bao Anh

## 📚 References
-   Kalehbasti, P. R., Nikolenko, L., & Rezaei, H. (2021). Airbnb price prediction using machine learning and sentiment analysis.
-   Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
-   Inside Airbnb. (2024). New York City Airbnb Data.

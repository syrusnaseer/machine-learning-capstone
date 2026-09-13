\# Medical Insurance Cost Classification — ML Capstone



An end-to-end machine learning classification project that predicts whether a person's medical insurance charge is \*\*High Cost\*\* or \*\*Low Cost\*\*, with model comparison, SHAP/LIME explainability, a Streamlit application, and Docker deployment.



\## Project Overview



This project was developed as a Machine Learning Capstone Project covering the complete ML workflow:



\* Data exploration and cleaning

\* Target engineering

\* Feature engineering and preprocessing

\* Classification model comparison

\* Final model selection

\* SHAP explainability

\* LIME explainability

\* Streamlit deployment

\* Docker containerization



\## Dataset



The project uses the \*\*Medical Cost Personal Dataset\*\* (`insurance.csv`) from a previous internship task.



The original dataset is a regression dataset containing medical insurance charges. For this capstone, it was adapted into a binary classification problem.



\### Dataset Size



\* \*\*Rows:\*\* 1,338

\* \*\*Original columns:\*\* 7

\* \*\*Target threshold:\*\* 9,382.033



A new target variable called `High\_Cost` was created:



\* `1` → High Cost

\* `0` → Low Cost



The original `charges` column is excluded from the model features because it is used to define the target. This prevents target leakage.



\### Input Features



\* `age`

\* `sex`

\* `bmi`

\* `children`

\* `smoker`

\* `region`



\## Data Preprocessing



Numerical features were standardized using `StandardScaler`.



Categorical features were encoded using `OneHotEncoder` with unknown-category handling.



The dataset was divided into training and testing sets using:



\* Test size: 20%

\* Random state: 42

\* Stratified split



\## Models Compared



Five classification approaches were trained and evaluated on the same train/test split:



1\. Logistic Regression

2\. Random Forest

3\. Decision Tree

4\. XGBoost

5\. Stacked Ensemble



\### Model Comparison



| Model               |  Precision | Recall |         F1 |    ROC-AUC |

| ------------------- | ---------: | -----: | ---------: | ---------: |

| Logistic Regression |     0.8849 | 0.9179 |     0.9011 |     0.9425 |

| Random Forest       |     0.9758 | 0.9030 |     0.9380 |     0.9490 |

| Decision Tree       |     0.8806 | 0.8806 |     0.8806 |     0.8806 |

| \*\*XGBoost\*\*         | \*\*0.9837\*\* | 0.9030 | \*\*0.9416\*\* | \*\*0.9505\*\* |

| Stacked Ensemble    |     0.9606 | 0.9104 |     0.9349 |     0.9466 |



\## Final Model



\*\*XGBoost\*\* was selected as the final model.



It achieved:



\* \*\*Precision:\*\* 0.9837

\* \*\*Recall:\*\* 0.9030

\* \*\*F1 Score:\*\* 0.9416

\* \*\*ROC-AUC:\*\* 0.9505



XGBoost provided the best overall F1 score and ROC-AUC, while also achieving the highest precision among the evaluated models.



The slightly lower recall compared with Logistic Regression was considered as part of the precision-recall tradeoff.



\## Explainability



Two explainability techniques were implemented.



\### SHAP



SHAP was used for both global and local explanations.



The project includes:



\* Global SHAP feature importance

\* Three local SHAP explanations

\* SHAP explanations integrated into the Streamlit application



SHAP values indicate how individual features influence a prediction toward High Cost or Low Cost.



\### LIME



LIME was used to explain individual predictions.



Three individual LIME explanations were generated and saved in the `explainability` directory.



\## Streamlit Application



The project includes an interactive Streamlit application.



Users can enter:



\* Age

\* Sex

\* BMI

\* Number of children

\* Smoker status

\* Region



The application provides:



\* High Cost / Low Cost prediction

\* High Cost probability

\* SHAP-based explanation

\* Input summary



\### Run Locally



Activate the virtual environment and run:



```powershell

python -m streamlit run app/app.py

```



The application will be available at:



```text

http://localhost:8501

```



\## Docker Deployment



The application is containerized using Docker.



\### Build the Image



```powershell

docker build -t medical-insurance-classifier .

```



\### Run the Container



```powershell

docker run -p 8501:8501 medical-insurance-classifier

```



The application can then be accessed at:



```text

http://localhost:8501

```



\### Docker Hub



The final Docker image is published as:



```text

syrusnaseer/medical-insurance-classifier:latest

```



\## Project Structure



```text

machine-learning-capstone/

│

├── app/

│   └── app.py

│

├── data/

│   └── insurance.csv

│

├── explainability/

│   ├── shap\_analysis.py

│   ├── shap\_local.py

│   ├── shap\_global\_summary.png

│   ├── shap\_local\_patient\_1.png

│   ├── shap\_local\_patient\_2.png

│   ├── shap\_local\_patient\_3.png

│   ├── lime\_analysis.py

│   ├── lime\_patient\_1.png

│   ├── lime\_patient\_2.png

│   └── lime\_patient\_3.png

│

├── models/

│   ├── final\_model.pkl

│   └── model\_comparison.csv

│

├── notebooks/

│   └── machine\_learning\_capstone.ipynb

│

├── Dockerfile

├── requirements.txt

├── model\_comparison.py

├── retrain\_model.py

└── README.md

```



\## Main Deliverables



\* Documented ML notebook

\* Cleaned and prepared dataset

\* Five-model comparison

\* Final XGBoost model

\* Model comparison CSV

\* Global SHAP analysis

\* Three local SHAP explanations

\* Three LIME explanations

\* Streamlit prediction application

\* Dockerized application

\* Docker Hub image

\* Final project report



\## Limitations



This project uses a medical insurance cost dataset originally designed for regression. The capstone therefore adapts it into a binary classification problem using the median insurance charge as the classification threshold.



The model should be considered an educational machine learning project and not a medical or financial decision-making system.



\## Conclusion



The project demonstrates a complete machine learning pipeline from data preparation and feature engineering through model comparison, explainability, and deployment.



Among the five evaluated models, \*\*XGBoost\*\* provided the strongest overall performance and was selected as the final classification model. SHAP and LIME were incorporated to make individual and global model behavior more interpretable, while Streamlit and Docker provided a deployable application.




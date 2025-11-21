Project Overview:

This data science project aimed to develop and evaluate machine learning models to predict the severity of traffic accidents. The core objective was to identify key risk factors that lead to severe outcomes, providing actionable insights to improve road safety. The project involved the end-to-end process of data preprocessing, feature engineering, model training, and evaluation, culminating in a proof-of-concept that demonstrates the potential of data-driven approaches in public safety policy.

Methodology & Technical Execution:

Custom Severity Metric: Introduced a novel Accident Severity Index (ASI), a continuous metric calculated from fatalities and injuries, which was then discretized into Low, Medium, and High severity classes for classification tasks.

Data Preprocessing & Feature Engineering: Consolidated multiple datasets (accident, vehicle, driver) and performed comprehensive preprocessing, including handling missing values, encoding categorical variables, and normalizing features. A key innovation was the creation of composite features (e.g., Speed_Road_Combo, Light_Road_Combo) to capture complex interactions in the accident environment.

Feature Selection: Utilized Pearson Correlation for continuous features and Mutual Information for categorical features to identify the most predictive variables, validating the importance of the engineered composite features.

Model Training & Evaluation: Implemented and compared two classification models:

K-Nearest Neighbours (KNN): Achieved higher overall accuracy (87.4%) but failed completely to identify high-severity accidents, indicating a limitation with imbalanced data.

Decision Tree: Achieved lower overall accuracy (59.7%) but demonstrated a critical ability to identify high-severity accidents (72% recall), making it a more useful model for a safety-focused application despite its low precision.

Key Findings & Insights:

The analysis successfully identified the most influential factors affecting accident severity. The Decision Tree's feature importance analysis revealed that Speed Zone, Light Condition, and Road Geometry were the top predictors, providing clear, data-backed evidence for targeted interventions.

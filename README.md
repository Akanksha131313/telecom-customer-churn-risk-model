**telecom-customer-churn-risk-model :** 

**Predict & Retain High-Risk Customers ->**

**Why it matters -**

• Helps telecoms reduce revenue loss by proactively identifying high-risk churn customers and understanding the top drivers behind churn.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

**Key Metrics & Business Insights -**

• Churn rate: ~49% → significant revenue risk

• Short-tenure churn (<12 months): ~48% → early-stage retention critical

• Top drivers: PaymentMethod, TotalCharges, TechSupport → focus areas for retention campaigns

• Model accuracy: 0.37 (limited due to small dataset & class imbalance; insights emphasize actionable trends rather than raw score)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Impact on Business -**

• Prioritize retention campaigns for high-risk customers (~49% churn)

• Target short-tenure customers to reduce early churn (~48%)

• Leverage top churn drivers to refine offers, support, and engagement strategies

• Enable data-driven decision making for customer retention initiatives

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

**Project Structure -**

• src/       → Python modules: 01_data_processing, 02_model_evaluation, 03_model_training

• data/      → telecom_churn_dataset_utf8.csv

• notebook/  → Interactive Colab/Jupyter analysis

• requirements.txt → Dependencies for reproducibility

• README.md  → Project overview & instructions

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

**Quick Start -**

1. Place dataset → data/telecom_churn_dataset_utf8.csv

2. Run 01_data_processing.py → clean & prepare data

3. Run 03_model_training.py → train model & print dynamic business insights

4. Optional: 02_model_evaluation.py → visualize metrics & confusion matrix

5. Open notebook → notebook/telecom_churn_analysis.ipynb for interactive analysis

------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Keywords for ATS-**

Telecom, churn prediction, Logistic Regression, feature importance, retention strategy, predictive model, classification report, confusion matrix, customer lifetime value


Optimized for recruiter quick scan and ATS compliance.
Emphasizes business impact, actionable insights, and practical application of predictive modeling.

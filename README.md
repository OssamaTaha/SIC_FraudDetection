# SIC_FraudDetection

# Financial Fraud Detection Using Batch Processing: Project Documentation

## Overview
This document provides a comprehensive guide for Project 2: Financial Fraud Detection Using Batch Processing. The project focuses on developing a system to detect fraudulent transactions in financial datasets using batch processing techniques. The workflow includes data ingestion from a relational database into Hadoop, processing with Apache Spark, querying with Hive, applying machine learning for fraud detection, and visualizing results with a business intelligence (BI) tool.

## Key Tasks

1. **Data Ingestion**: Import financial transaction data into Hadoop HDFS using Sqoop from a MySQL database.
2. **Data Processing & Transformation**: Use Apache Spark for data processing and transformation.
3. **Querying**: Query and analyze the data using Hive.
4. **Machine Learning**: Apply machine learning algorithms to detect fraudulent activities.
5. **Visualization & Reporting**: Visualize the detection results using a BI tool and generate insightful reports.

## Technologies

- **Sqoop**: Data ingestion from MySQL to Hadoop.
- **Hadoop**: Storage and management of large volumes of transaction data.
- **Apache Spark**: Data processing and transformation.
- **Hive**: Querying and analyzing data.
- **BI Tool**: Visualization and reporting.
- **ML Algorithms**: Decision trees, SVM, and other algorithms for fraud detection.

## Data Ingestion

### 1. Data Preparation

#### 1.1 Analyzing Column Types

```python
import pandas as pd

# Load CSV file
df = pd.read_csv('fraudTrain.csv')

# Analyze columns and suggest MySQL data types
def suggest_mysql_dtype(df):
    type_mapping = {
        'int64': 'INT',
        'float64': 'FLOAT',
        'object': 'VARCHAR(255)',
        'bool': 'TINYINT(1)',
        'datetime64[ns]': 'DATETIME'
    }

    for column, dtype in df.dtypes.items():
        mysql_dtype = type_mapping.get(str(dtype), 'VARCHAR(255)')
        print(f"Column: {column}, Pandas Type: {dtype}, Suggested MySQL Type: {mysql_dtype}")

suggest_mysql_dtype(df)
```

**Resulting Data Types:**

| Column                    | Pandas Type | Suggested MySQL Type |
|---------------------------|-------------|----------------------|
| Unnamed: 0                 | int64        | INT                  |
| trans_date_trans_time     | object       | VARCHAR(255)         |
| cc_num                    | int64        | INT                  |
| merchant                  | object       | VARCHAR(255)         |
| category                  | object       | VARCHAR(255)         |
| amt                       | float64      | FLOAT                |
| first                     | object       | VARCHAR(255)         |
| last                      | object       | VARCHAR(255)         |
| gender                    | object       | VARCHAR(255)         |
| street                    | object       | VARCHAR(255)         |
| city                      | object       | VARCHAR(255)         |
| state                     | object       | VARCHAR(255)         |
| zip                       | int64        | INT                  |
| lat                       | float64      | FLOAT                |
| long                      | float64      | FLOAT                |
| city_pop                  | int64        | INT                  |
| job                       | object       | VARCHAR(255)         |
| dob                       | object       | VARCHAR(255)         |
| trans_num                 | object       | VARCHAR(255)         |
| unix_time                 | int64        | INT                  |
| merch_lat                 | float64      | FLOAT                |
| merch_long                | float64      | FLOAT                |
| is_fraud                  | int64        | INT                  |

#### 1.2 Modifying the CSV File

```python
# Rename the first column
df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)

# Save the modified CSV file
df.to_csv('fraudTrain_modified.csv', index=False)
```

**Directory Setup Command:**

```bash
mkdir -p /home/student/fraudDetection/Dataset
```

Drag and drop the file `fraudTrain_modified.csv` to `/home/student/fraudDetection/Dataset`.

### 2. MySQL Database Setup

#### 2.1 Log in to MySQL

```bash
mysql --user=student --password=student
```

#### 2.2 Create Database and Table

```sql
CREATE DATABASE financial_fraud;
USE financial_fraud;

CREATE TABLE transactions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    trans_date_trans_time VARCHAR(255),
    cc_num BIGINT,  
    merchant VARCHAR(255),
    category VARCHAR(255),
    amt FLOAT,
    first VARCHAR(255),
    last VARCHAR(255),
    gender VARCHAR(255),
    street VARCHAR(255),
    city VARCHAR(255),
    state VARCHAR(255),
    zip INT,
    lat FLOAT,
    lon FLOAT,  
    city_pop INT,
    job VARCHAR(255),
    dob DATE,  
    trans_num VARCHAR(255), 
    unix_time INT,
    merch_lat FLOAT,
    merch_long FLOAT, 
    is_fraud TINYINT(1)  
);
```

#### 2.3 Load Data into the Table

```sql
LOAD DATA LOCAL INFILE '/home/student/fraudDetection/Dataset/fraudTrain_modified.csv'
INTO TABLE transactions
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(id, trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender, street, city, state, zip, lat, lon, city_pop, job, dob, trans_num, unix_time, merch_lat, merch_long, is_fraud);
```

### 3. Import Data from MySQL to Hadoop

#### 3.1 Sqoop Import Command

```bash
sqoop import \
    --connect jdbc:mysql://localhost/financial_fraud \
    --username student \
    --password student \
    --table transactions \
    --split-by id \
    --incremental append \
    --check-column id \
    --last-value 0 \
    --as-parquetfile \
    --verbose
```

## Data Processing & Transformation

### 1. Importing Libraries & Necessary Steps

#### 1.1 Import Libraries

```python
from pyspark.sql import SparkSession
```

#### 1.2 Reading Parquet File

```python
spark = SparkSession.builder.appName("FraudDetection").getOrCreate()
df = spark.read.parquet("/user/student/transactions_new/updated_fraud_data_parquet")
```

#### 1.3 Schema and Data Type Corrections

**Found Issues:**

- **DOB**: Change from `long` to `date`.
- **UNIX_TIME**: Change from `int` to `timestamp`.
- **IS_FRAUD**: Change from `boolean` to `integer`.

```python
from pyspark.sql.functions import col, to_date, from_unixtime

df = df.withColumn("dob", to_date(col("dob"), "yyyy-MM-dd"))
df = df.withColumn("unix_time", from_unixtime(col("unix_time")))
df = df.withColumn("is_fraud", col("is_fraud").cast("int"))
```

#### 2. Transformations

**Spark DataFrame to Pandas DataFrame**

```python
pandas_df = df.toPandas()
```

### 3. Data Cleaning

#### 3.1 Check for Nulls

```python
print(df.isNull().sum())
```

#### 3.2 Check for Duplicates

```python
print(df.count() - df.dropDuplicates().count())
```

#### 3.3 Exploring Numeric Data with Box Plots

```python
import matplotlib.pyplot as plt

# Example for amount
plt.boxplot(pandas_df['amt'])
plt.title('Box Plot of Transaction Amounts')
plt.show()
```

**Conclusion:** 
The dataset does not contain nulls. Duplicates are expected and acceptable. No unusual outliers were found.

### 4. Querying

#### 4.1 Total Fraudulent Transactions by Merchant

```sql
SELECT merchant, COUNT(*) as fraudulent_count
FROM transactions
WHERE is_fraud = 1
GROUP BY merchant
ORDER BY fraudulent_count DESC;
```

#### 4.2 Top Cities by Number of Fraudulent Transactions

```sql
SELECT city, COUNT(*) as fraudulent_count
FROM transactions
WHERE is_fraud = 1
GROUP BY city
ORDER BY fraudulent_count DESC;
```

#### 4.3 Fraudulent Transactions by Time of Day

```sql
SELECT HOUR(trans_date_trans_time) as hour, COUNT(*) as fraudulent_count
FROM transactions
WHERE is_fraud = 1
GROUP BY hour
ORDER BY hour;
```

#### 4.4 Average Transaction Amount for Fraudulent vs Non-Fraudulent Transactions

```sql
SELECT is_fraud, AVG(amt) as avg_amount
FROM transactions
GROUP BY is_fraud;
```

#### 4.5 Fraud Rate by Category

```sql
SELECT category, SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) / COUNT(*) as fraud_rate
FROM transactions
GROUP BY category;
```

#### 4.6 Find the Most Common Job Among Fraudsters

```sql
SELECT job, COUNT(*) as count
FROM transactions
WHERE is_fraud = 1
GROUP BY job
ORDER BY count DESC;
```

#### 4.7 Find if Fraud is Affected by Gender

```sql
SELECT gender, COUNT(*) as count
FROM transactions
WHERE is_fraud = 1
GROUP BY gender;
```

**Conclusion:**
- **Top Fraudulent Cities**:

 Cities with the highest number of fraud cases are highlighted.
- **Fraudulent Transactions by Time of Day**: Peak hours for fraud are identified.
- **Fraud Rates by Category**: Categories with higher fraud rates are listed.
- **Average Transaction Amount**: Comparison between fraudulent and non-fraudulent transactions.
- **Most Common Job Among Fraudsters**: Jobs that are common among fraudulent transactions.
- **Gender Distribution**: Fraud distribution by gender.

### 5. Machine Learning

Here's a README file that explains the code provided:

---

# Fraud Detection Model

## Overview

This project involves training and evaluating a Logistic Regression model for fraud detection using Apache Spark. The dataset used contains transaction information, and the goal is to predict fraudulent transactions. The process includes data preprocessing, model training, evaluation, and visualization of results.

## Setup

Ensure you have the following libraries installed in your Spark environment:
- `pyspark`
- `matplotlib`
- `seaborn`
- `sklearn`
- `pandas`

## Code Breakdown

### 1. Import Libraries

```python
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col, when
from pyspark.sql.types import StructType, ArrayType
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import pandas as pd
```

### 2. Model Training

#### a. Load Data

```python
# Define the path to the CSV file in HDFS
csv_path = "hdfs:///user/student/transactions_new/updated_fraud_data_csv"

# Read the CSV file into a DataFrame
df = spark.read.csv(csv_path, header=True, inferSchema=True)
```

#### b. Add Transaction Hour Column

```python
# Added a new column to calculate the transaction hour
transformed_df = df.withColumn("trans_hour", col("trans_date_trans_time").substr(12, 2).cast('int'))
```

#### c. Categorical Feature Encoding

```python
# List of categorical columns
categorical_columns = ['merchant', 'category', 'gender', 'city', 'state', 'job']

# Index and encode categorical columns
indexers = []
encoders = []

for col_name in categorical_columns:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_index")
    encoder = OneHotEncoder(inputCols=[col_name + "_index"], outputCols=[col_name + "_encoded"])
    
    indexers.append(indexer)
    encoders.append(encoder)
    
    transformed_df = indexer.fit(transformed_df).transform(transformed_df)
    transformed_df = encoder.fit(transformed_df).transform(transformed_df)
```

#### d. Feature Engineering

```python
# Select feature columns, including encoded categorical features
feature_columns = ['amt', 'trans_hour', 'city_pop', 'lat', 'lon', 'merch_lat', 'merch_long'] \
    + [col + "_encoded" for col in categorical_columns]

# Apply VectorAssembler
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
ml_data = assembler.transform(transformed_df)
```

#### e. Handling Class Imbalance

```python
# Handling class imbalance by adding class weights
fraud_ratio = ml_data.filter(col('is_fraud') == 1).count() / ml_data.count()
ml_data = ml_data.withColumn("class_weight", when(col("is_fraud") == 1, 1 / fraud_ratio).otherwise(1.0))
```

#### f. Data Splitting

```python
# Split the data into 50% train and 50% test
train_data, test_data = ml_data.randomSplit([0.5, 0.5])
```

#### g. Train Logistic Regression Model

```python
# Initialize and train the Logistic Regression model with regularization
lr = LogisticRegression(featuresCol='features', labelCol='is_fraud', weightCol="class_weight", regParam=0.01, elasticNetParam=0.8)
lr_model = lr.fit(train_data)
```

#### h. Make Predictions

```python
# Make predictions on the test data
lr_predictions = lr_model.transform(test_data)
```

### 3. Model Evaluation

#### a. Metrics Calculation

```python
# Initialize evaluators
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="is_fraud", predictionCol="prediction", metricName="accuracy")
precision_evaluator = MulticlassClassificationEvaluator(labelCol="is_fraud", predictionCol="prediction", metricName="precisionByLabel")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="is_fraud", predictionCol="prediction", metricName="recallByLabel")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="is_fraud", predictionCol="prediction", metricName="f1")
roc_evaluator = BinaryClassificationEvaluator(labelCol="is_fraud", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

# Calculate metrics
accuracy = accuracy_evaluator.evaluate(lr_predictions)
precision = precision_evaluator.evaluate(lr_predictions, {precision_evaluator.metricLabel: 1.0})  # for class 1 (fraud)
recall = recall_evaluator.evaluate(lr_predictions, {recall_evaluator.metricLabel: 1.0})  # for class 1 (fraud)
f1_score = f1_evaluator.evaluate(lr_predictions)
roc_auc = roc_evaluator.evaluate(lr_predictions)

# Print metrics
print(f"Logistic Regression Accuracy: {accuracy}")
print(f"Logistic Regression Precision (Fraud Class): {precision}")
print(f"Logistic Regression Recall (Fraud Class): {recall}")
print(f"Logistic Regression F1 Score: {f1_score}")
print(f"Logistic Regression ROC-AUC: {roc_auc}")
```

#### b. Confusion Matrix

```python
# Create and show the confusion matrix
confusion_matrix_df = lr_predictions.groupBy("is_fraud", "prediction").count()

# Convert Spark DataFrame to Pandas DataFrame for confusion matrix
predictions_and_labels = lr_predictions.select(col("prediction"), col("is_fraud")).rdd
prediction_df = predictions_and_labels.toDF(["prediction", "label"]).toPandas()

# Compute confusion matrix
conf_matrix = sk_confusion_matrix(prediction_df["label"], prediction_df["prediction"])

# Create a DataFrame for visualization
conf_matrix_df = pd.DataFrame(conf_matrix, index=["True Negative", "False Positive"], columns=["False Negative", "True Positive"])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig("confusion_matrix_plot.png")  # Save the plot as an image file
plt.show()
```

### 4. Save Results

#### a. Save Confusion Matrix to HDFS

```python
# Convert confusion matrix DataFrame to a Spark DataFrame
conf_matrix_spark_df = spark.createDataFrame(conf_matrix_df.reset_index())
# Save confusion matrix to CSV in a single partition
conf_matrix_spark_df.repartition(1).write.option("header", "true").mode("overwrite").csv("hdfs:///user/student/transactions_new/confusion_matrix.csv")
```

#### b. Save Evaluation Metrics to HDFS

```python
# Save evaluation metrics to CSV in a single partition (HDFS)
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision (Fraud Class)", "Recall (Fraud Class)", "F1 Score", "ROC-AUC"],
    "Value": [accuracy, precision, recall, f1_score, roc_auc]
})

# Convert metrics DataFrame to Spark DataFrame
metrics_spark_df = spark.createDataFrame(metrics_df)

# Save evaluation metrics to CSV in a single partition
metrics_spark_df.repartition(1).write.option("header", "true").mode("overwrite").csv("hdfs:///user/student/transactions_new/metrics.csv")
```

## Conclusion

This code provides a comprehensive pipeline for building, evaluating, and saving the results of a Logistic Regression model for fraud detection. It includes steps for data preprocessing, feature engineering, model training, evaluation, and visualization. Ensure to adjust paths and configurations as needed for your environment.

--- 

Feel free to adjust the paths and any other details according to your specific setup.
### 6. Visualization & Reporting

**Visualization Examples**

#### 6.1 Fraud Rate by Category

**Using Tableau or Power BI:**

- Create a bar chart with categories on the x-axis and fraud rate on the y-axis.

#### 6.2 Top Fraudulent Merchants

**Using Tableau or Power BI:**

- Create a bar chart with merchants on the x-axis and the count of fraudulent transactions on the y-axis.

**Reports**

Generate reports summarizing findings, model performance, and key insights for stakeholders.

---

This documentation provides a structured approach for developing a financial fraud detection system using batch processing techniques. Ensure to adapt and expand on the provided examples and code snippets as per your project requirements.

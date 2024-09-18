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

**Building a Model**

#### 5.1 Feature Selection

Select relevant features such as amount, category, city, and others based on data exploration.

#### 5.2 Building the Model

```python
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

# Feature Assembly
assembler = VectorAssembler(inputCols=["amt", "lat", "lon"], outputCol="features")

# Random Forest Classifier
rf = RandomForestClassifier(labelCol="is_fraud", featuresCol="features")

# Pipeline
pipeline = Pipeline(stages=[assembler, rf])

# Fit Model
model = pipeline.fit(df)
```

#### 5.3 Model Evaluation

```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Predictions
predictions = model.transform(df)

# Evaluator
evaluator = BinaryClassificationEvaluator(labelCol="is_fraud")
print("Area Under ROC: ", evaluator.evaluate(predictions))
```

**Conclusion:**
The model's performance is evaluated based on metrics like AUC-ROC.

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

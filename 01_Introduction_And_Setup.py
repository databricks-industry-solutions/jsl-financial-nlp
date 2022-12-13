# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction
# MAGIC In this series of notebooks, we are going to analyze a 10K filing, obtained from US Security Exchange Commission's Edgar database, and create a financial knowledge graph with information from it, including registry information...
# MAGIC - `Official name of the company used in Edgar`
# MAGIC - `Other identification numbers, as CIK, SIC (Sector Code), IRS`
# MAGIC - `Stock information: Stock Market, Title Class, Class values, Trading Symbol, etc.`
# MAGIC - `Registry information (addresses, phone numbers, state, etc)`
# MAGIC 
# MAGIC ... information about **other companies**...
# MAGIC - `Competitors`
# MAGIC - `Companies in the Supply Chain / mentioned in the filing to be in the ecosystem of the company`
# MAGIC 
# MAGIC ... **people** ...
# MAGIC - `Current C-level managers`
# MAGIC - `Past C-level managers mentioned in the 10K filing (usually founders and co-founders)`
# MAGIC - `and their past experiences, if mentioned!`
# MAGIC 
# MAGIC Also, we will apply **Data Augmentation** using **offline John Snow Labs data sources** (`ChunkMappers`) to map the name of the company to information we know about it and John Snow Labs updates at a quartely basis, including:
# MAGIC - `Former Names`
# MAGIC - `Year of the change`

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/JohnSnowLabs/spark-nlp-workshop/raw/master/tutorials/Certification_Trainings_JSL/Finance/data/img10.png" alt="drawing" width="800"/>

# COMMAND ----------

# MAGIC %md
# MAGIC The Financial NLP library already comes preinstalled with John Snow Labs - Databricks Partner Connect, available [here](https://docs.databricks.com/partners/ml/john-snow-labs.html).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %md
# MAGIC Let's check `johnsnowlabs` library is installed

# COMMAND ----------

import johnsnowlabs
print(f"Spark NLP Licensed: {johnsnowlabs.settings.raw_version_jsl_lib}")
print(f"Spark NLP Open Source: {johnsnowlabs.settings.raw_version_nlp}")
print(f"Spark NLP PySpark: {johnsnowlabs.settings.raw_version_pyspark}")

# COMMAND ----------

# MAGIC %md
# MAGIC Let's install the graph and visualization libraries

# COMMAND ----------

!pip install -q networkx==2.5 decorator==5.0.9 plotly==5.1.0

# COMMAND ----------

# MAGIC %md
# MAGIC Let's import them and reload, in case other versions of the libraries were present

# COMMAND ----------

import importlib
import networkx as nx
import decorator as dc
import plotly

importlib.reload(nx)
importlib.reload(dc)

# COMMAND ----------

print(f"NetworkX version: {nx.__version__}")
print(f"Decorator version: {dc.__version__}")
print(f"Plotly version: {plotly.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC # You can proceed to 02 - Analysis!

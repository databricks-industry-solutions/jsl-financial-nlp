# Databricks notebook source
# MAGIC %md
# MAGIC # Analysis
# MAGIC During this process of analysis, let's get a sample of a 10-K filing from the internet, visualize it and understand the company information there

# COMMAND ----------

# MAGIC %md
# MAGIC # Sample Texts from Cadence Design System
# MAGIC Examples taken from publicly available information about Cadence in SEC's Edgar database [here](https://www.sec.gov/Archives/edgar/data/813672/000081367222000012/cdns-20220101.htm) and [Wikipedia](https://en.wikipedia.org/wiki/Cadence_Design_Systems)

# COMMAND ----------

!wget https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings_JSL/Finance/data/cdns-20220101.html.txt

# COMMAND ----------

# MAGIC %md
# MAGIC **It's a 10K filing..**

# COMMAND ----------

with open('cdns-20220101.html.txt', 'r') as f:
  cadence_sec10k = f.read()
print(cadence_sec10k[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC **The company is Cadence Design System, Inc**

# COMMAND ----------

with open('cdns-20220101.html.txt', 'r') as f:
  cadence_sec10k = f.read()
print(cadence_sec10k[:700])

# COMMAND ----------

# MAGIC %md
# MAGIC # Let's split the document into pages

# COMMAND ----------

pages = [x for x in cadence_sec10k.split("Table of Contents") if x.strip() != '']
print(f"Number of pages: {len(pages)}")

# COMMAND ----------

# MAGIC %md
# MAGIC The first page has the usual **10-K summary** information, which is very useful

# COMMAND ----------

print(pages[0])

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Finance/data/10k_image.png?raw=true"/>

# COMMAND ----------

# MAGIC %md
# MAGIC Some pages have information about **acquisitions and subsidiaries**...

# COMMAND ----------

print("...\n" + pages[67][400:2100] + "...\n")

# COMMAND ----------

# MAGIC %md
# MAGIC We also have some information about **People (C-level management)** across the document...

# COMMAND ----------

print("...\n" + pages[4][2000:] + "...\n")

# COMMAND ----------

# MAGIC %md
# MAGIC # Continue you analysis
# MAGIC Make sure you understand the contents of a 10-K and what information can be extracted. In the following notebooks we are going to extract all of that information using Finance NLP

# COMMAND ----------

# MAGIC %md
# MAGIC # You are ready to proceed to the 03 Entity Extraction notebook!

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In next and the following notebooks you will use Finance NLP to extract information, more specifically:
# MAGIC - `NER`: To extract financial entities from the summary page, organizations, people and roles across the document;
# MAGIC - `Normalization` to obtain the official name of the company (and any other former name) in Edgar, and `ChunkMappers` to retrieve all the registry information available in Edgar for that company;
# MAGIC - `Relation Extraction`: to obtain acquisitions and subsidiaries mentioned, and also people and their roles in companies;
# MAGIC - ... and much more!

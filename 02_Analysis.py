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
# MAGIC ## Let's use `SentenceDetector` as a Page Splitter
# MAGIC Using `Table of Contents`, which is present at the end of each page as a marker of new page

# COMMAND ----------

from johnsnowlabs import nlp, finance

document_assembler = nlp.DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

sentence_detector = nlp.SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("pages")\
    .setCustomBounds(["Table of Contents"])\
    .setUseCustomBoundsOnly(True)

nlp_pipeline = nlp.Pipeline(stages=[
    document_assembler, 
    sentence_detector])

# COMMAND ----------

from johnsnowlabs.nlp import LightPipeline

empty_data = spark.createDataFrame([[""]]).toDF("text")
sentence_splitting_pipe = nlp_pipeline.fit(empty_data)
sentence_splitting_lightpipe = LightPipeline(sentence_splitting_pipe)

# COMMAND ----------

res = sentence_splitting_lightpipe.annotate(cadence_sec10k)
pages = res['pages']
pages = [p for p in pages if p.strip() != ''] # We remove empty pages

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
# MAGIC ## Using Text Classification to find Relevant Parts of the Document: 10K Summary
# MAGIC In this case, we know page 0 is always the page with summary information about the company. However, let's suppose we don't know it. We can use `ClassifierDL` to do Text Classification, in this case, at `Page` level.
# MAGIC 
# MAGIC To check the SEC 10K Summary page, we have a specific model called `"finclf_form_10k_summary_item"`

# COMMAND ----------

# MAGIC %run "./aux_pipeline_functions"

# COMMAND ----------

classification_pipeline = get_text_classification_pipeline('finclf_form_10k_summary_item')

empty_data = spark.createDataFrame([[""]]).toDF("text")

fit_classification_pipeline = classification_pipeline.fit(empty_data)

# COMMAND ----------

from johnsnowlabs.nlp import LightPipeline

light_classification_pipeline = LightPipeline(fit_classification_pipeline)
result = light_classification_pipeline.annotate(pages[0])

# COMMAND ----------

result['category']

# COMMAND ----------

[x['category'] for x in light_classification_pipeline.annotate([pages[1], pages[2], pages[70], pages[80]])]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Text Classification to find Relevant Parts of the Document: Acquisitions and Subsidiaries
# MAGIC To check the SEC 10K Summary page, we have a specific model called `"finclf_acquisitions_item"`
# MAGIC 
# MAGIC Let's send some pages and check which one(s) contain that information. In a real case, you could send all the pages to the model, but here for time saving purposes, we will show just a subset.

# COMMAND ----------

candidates = [[pages[0]], [pages[1]], [pages[35]], [pages[50]], [pages[67]]] # Some examples

# COMMAND ----------

classification_pipeline = get_text_classification_pipeline('finclf_acquisitions_item')
df = spark.createDataFrame(candidates).toDF("text")
model = classification_pipeline.fit(df)
result = model.transform(df)

# COMMAND ----------

result.select('category.result').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Text Classification to find Relevant Parts of the Document: About Management and their work experience
# MAGIC To check the SEC 10K Summary page, we have a specific model called `"finclf_work_experience_item"`
# MAGIC 
# MAGIC Let's send some pages and check which one(s) contain that information. In a real case, you could send all the pages to the model, but here for time saving purposes, we will show just a subset.

# COMMAND ----------

candidates = [[pages[4]], [pages[84]], [pages[85]], [pages[86]], [pages[87]]]

# COMMAND ----------

classification_pipeline = get_text_classification_pipeline('finclf_work_experience_item')
df = spark.createDataFrame(candidates).toDF("text")
model = classification_pipeline.fit(df)
result = model.transform(df)
result.select('category.result').show()

# COMMAND ----------

# MAGIC %md
# MAGIC **We have some Work Experience in page 86. However, there is 1 sentence hidden in page 4, which is also very relevant.**
# MAGIC 
# MAGIC However, the model returned `other`. Why?

# COMMAND ----------

pages[4]

# COMMAND ----------

# MAGIC %md
# MAGIC Exploring the page we understand there is a lot of texts about something else which got into the same page. Sometimes, going into a smaller detail may be necessary.
# MAGIC 
# MAGIC Let's see what happens if we get `paragraphs` instead of `pages.`

# COMMAND ----------

from johnsnowlabs import nlp, finance

document_assembler = nlp.DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

sentence_detector = nlp.SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("paragraphs")

nlp_pipeline = nlp.Pipeline(stages=[
    document_assembler, 
    sentence_detector])

# COMMAND ----------

from johnsnowlabs.nlp import LightPipeline

empty_data = spark.createDataFrame([[""]]).toDF("text")
sentence_splitting_pipe = nlp_pipeline.fit(empty_data)
sentence_splitting_lightpipe = LightPipeline(sentence_splitting_pipe)

# COMMAND ----------

res = sentence_splitting_lightpipe.annotate(pages[4])
paragraphs = res['paragraphs']
paragraphs = [p for p in paragraphs if p.strip() != ''] # We remove empty pages

# COMMAND ----------

paragraphs

# COMMAND ----------

candidates = [[x] for x in paragraphs]

# COMMAND ----------

classification_pipeline = get_text_classification_pipeline('finclf_work_experience_item')
df = spark.createDataFrame(candidates).toDF("text")
model = classification_pipeline.fit(df)
result = model.transform(df)
result.select('category.result').show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Here we are, if we split in smaller detail (paragraphs, lines), we can found more information than just at page level!**
# MAGIC 
# MAGIC This is because information in Embeddings gets deluted the bigger the text is. Also, there are some text restrictions (512 tokens in Bert)

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

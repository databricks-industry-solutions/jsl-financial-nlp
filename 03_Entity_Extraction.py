# Databricks notebook source
# MAGIC %md
# MAGIC # Entity Extraction
# MAGIC Let's proceed to extract the entities we know from previous steps (and for our knowledge of 10K or 10Q filings) that are available in our document.

# COMMAND ----------

from johnsnowlabs import nlp, finance, viz

import os
import sys
import time
import json
import functools 
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import spatial

# COMMAND ----------

# MAGIC %md
# MAGIC ### Auxiliary Visualization functions 
# MAGIC We will use [NetworkX](https://networkx.org/) to store the graph and [Plotly](https://plotly.com/) to visualize it.
# MAGIC 
# MAGIC These functions will:
# MAGIC - Use Plotly to visualize a NetworkX graph
# MAGIC - Display relations in a dataframe

# COMMAND ----------

# MAGIC %run "./aux_visualization_functions"

# COMMAND ----------

G = nx.Graph()

# COMMAND ----------

# MAGIC %md
# MAGIC # Auxiliary Pipeline functions
# MAGIC In an independent file, we save 2 common pipelines we will be used all over the document, to keep the notebooks clean:
# MAGIC - **a generic pipeline**: having `DocumentAssembler`, `SentenceDetector`, `Tokenizer` and `Financial Embeddings`;
# MAGIC - **a text classification pipeline**: having `DocumentAssembler`, `Sentence Embeddings (Universal Sentence Embedings)` and `ClassifierDL (Text Classification)`;

# COMMAND ----------

# MAGIC %run "./aux_pipeline_functions"

# COMMAND ----------

generic_base_pipeline = get_generic_base_pipeline()

# COMMAND ----------

# MAGIC %md
# MAGIC # Let's start
# MAGIC We read back our text file of 90 pages

# COMMAND ----------

#!wget https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings_JSL/Finance/data/cdns-20220101.html.txt

# COMMAND ----------

with open('cdns-20220101.html.txt', 'r') as f:
  cadence_sec10k = f.read()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's use `SentenceDetector` as a Page Splitter
# MAGIC Using `Table of Contents`, which is present at the end of each page as a marker of new page

# COMMAND ----------

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

print(pages[0])

# COMMAND ----------

# MAGIC %md 
# MAGIC ## NER: Named Entity Recognition on 10K Summary
# MAGIC Main component to carry out information extraction and extract entities from texts. 
# MAGIC 
# MAGIC This time we will use a model trained to extract many entities from 10K summaries.

# COMMAND ----------

summary_sample_text = pages[0]

# COMMAND ----------

ner_model_sec10k = finance.NerModel.pretrained("finner_sec_10k_summary", "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner_summary")

ner_converter_sec10k = nlp.NerConverter()\
    .setInputCols(["sentence","token","ner_summary"])\
    .setOutputCol("ner_chunk_sec10k")

summary_pipeline = nlp.Pipeline(stages=[
    generic_base_pipeline,
    ner_model_sec10k,
    ner_converter_sec10k
])

# COMMAND ----------

from johnsnowlabs.nlp import LightPipeline

ner_vis = viz.NerVisualizer()

empty_data = spark.createDataFrame([[""]]).toDF("text")

summary_model = summary_pipeline.fit(empty_data)

light_summary_model = LightPipeline(summary_model)

summary_results = light_summary_model.fullAnnotate(summary_sample_text)

# COMMAND ----------

summary_results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Results

# COMMAND ----------

for r in summary_results:
    displayHTML(ner_vis.display(r, label_col = "ner_chunk_sec10k", document_col = "document", return_html=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## First, let's extract the Organization from NER results

# COMMAND ----------

# MAGIC %md
# MAGIC We create a new graph

# COMMAND ----------

G.clear()
G.nodes()

# COMMAND ----------

ORG = next(filter(lambda x: x.metadata['entity']=='ORG', summary_results[0]['ner_chunk_sec10k'])).result
ORG

# COMMAND ----------

# MAGIC %md
# MAGIC We add our first node to the graph

# COMMAND ----------

# I add our main Organization in the center (x=0, y=0)
G.add_node(ORG, attr_dict={'entity': 'ORG'})

# COMMAND ----------

show_graph_in_plotly(G)

# COMMAND ----------

# MAGIC %md
# MAGIC Then, let's add all the summary information from SEC 10K filings (1st page) to that organization.
# MAGIC 
# MAGIC We can create nodes and add a relation to Cadence directly, since we know it's information of that company.

# COMMAND ----------

for i, r in enumerate(summary_results[0]['ner_chunk_sec10k']):
  text = r.result
  entity = r.metadata['entity']
  
  if entity == 'ORG':
    continue #Already added
  G.add_node(text, attr_dict={'entity': entity}),
  G.add_edge(ORG, text, attr_dict={'relation': 'has_' + entity.lower()})  

# COMMAND ----------

show_graph_in_plotly(G)

# COMMAND ----------

import pickle

# save graph object to file
pickle.dump(G, open('cadence.pickle', 'wb'))

# COMMAND ----------

# MAGIC %md
# MAGIC # Now you can proceed to 04 Normalization and Data Augmentation!

# Databricks notebook source
# MAGIC %md
# MAGIC # Financial Solution Accelerator: Drawing a Company Ecosystem Graph and Analyzing it with Embeddings
# MAGIC This accelerator will help you process Financial Annual Reports (10K filings) or even Wikipedia data about companies, using John Snow Labs Finance NLP **Named Entity Recognition, Relation Extraction and Assertion Status**, to extract the following information about companies:
# MAGIC - Information about the Company itself (`Trading Symbol`, `State`, `Address`, Contact Information) and other names the Company is known by (`alias`, `former name`).
# MAGIC - People (usually management and C-level) working in that company and their past experiences, including roles and companies
# MAGIC - `Acquisitions` events, including the acquisition dates. `Subsidiaries` mentioned.
# MAGIC - Other Companies mentioned in the report as `competitors`: we will also run a "Competitor check", to understand if another company is just in the ecosystem / supply chain of the company or it is really a competitor
# MAGIC - Temporality (`past`, `present`, `future`) and Certainty (`possible`) of events described, including `Forward-looking statements`.
# MAGIC 
# MAGIC Also, John Snow Labs provides with offline modules to check for Edgar database (**Entity Linking** to resolve an organization name to its official name and **Chunk Mappers** to map a normalized name to Edgar Database), which are quarterly updated. We will using them to retrieve the `official name of a company`, `former names`, `dates where names where changed`, etc.

# COMMAND ----------

# MAGIC %md
# MAGIC ## License
# MAGIC Copyright / License info of the notebook. Copyright [2021] the Notebook Authors. The source in this notebook is provided subject to the Apache 2.0 License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC ## Libraries
# MAGIC Library Name	Library License	Library License URL	Library Source URL
# MAGIC ...
# MAGIC 
# MAGIC ## Author
# MAGIC Databricks Inc.
# MAGIC John Snow Labs Inc.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Disclaimers
# MAGIC ....

# COMMAND ----------

# MAGIC %md
# MAGIC The final aim of this accelerator is to help you analyze companies information...

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/JohnSnowLabs/spark-nlp-workshop/raw/master/tutorials/Certification_Trainings_JSL/Finance/data/im1.png" alt="drawing" width="600"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ... create a graph...

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/JohnSnowLabs/spark-nlp-workshop/raw/master/tutorials/Certification_Trainings_JSL/Finance/data/img10.png" alt="drawing" width="800"/>

# COMMAND ----------

# MAGIC %md
# MAGIC # Get Started with Databricks Partner Connect with John Snow Labs
# MAGIC John Snow Labs Spark Finance NLP Library, integrated in Databricks.
# MAGIC Ask for your license [here](https://docs.databricks.com/integrations/ml/john-snow-labs.html)

# COMMAND ----------

# MAGIC %md
# MAGIC # Starting a session

# COMMAND ----------

# MAGIC %md
# MAGIC In Databricks you will already have a Spark session started for you. 
# MAGIC 
# MAGIC If it's not the case, you only need to do:
# MAGIC `jsl.start(license_json_path=[your_path_to_json_license])`

# COMMAND ----------

# MAGIC %pip install johnsnowlabs networkx plotly

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Imports

# COMMAND ----------

from johnsnowlabs import * 
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

import networkx as nx
G = nx.Graph()

# COMMAND ----------

import plotly.graph_objects as go
import random

def get_nodes_from_graph(graph, pos, node_color):
  """Extracts the nodes from a networkX dataframe in Plotly Scatterplot format"""
  node_x = []
  node_y = []
  texts = []
  hovers = []
  for node in graph.nodes():
    entity = graph.nodes[node]['attr_dict']['entity']
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    texts.append(node)
    hovers.append(entity)

  node_trace = go.Scatter(
    x=node_x, y=node_y, text=texts, hovertext=hovers,
    mode='markers+text',
    hoverinfo='text',
    marker=dict(
        color=node_color,
        size=40,
        line_width=2))
  
  return node_trace


def get_edges_from_graph(graph, pos, edge_color):
  """Extracts the edges from a networkX dataframe in Plotly Scatterplot format"""
  edge_x = []
  edge_y = []
  hovers = []
  xtext = []
  ytext = []
  for edge in graph.edges():
    relation = graph.edges[edge]['attr_dict']['relation']
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)
    hovers.append(relation)
    xtext.append((x0+x1)/2)
    ytext.append((y0+y1)/2)

  edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=2, color=edge_color),
    mode='lines')
  
  labels_trace = go.Scatter(x=xtext,y= ytext, mode='text',
                              textfont = {'color': edge_color},
                              marker_size=0.5,
                              text=hovers,
                              textposition='top center',
                              hovertemplate='weight: %{text}<extra></extra>')
  return edge_trace, labels_trace


def show_graph_in_plotly(graph, node_color='white', edge_color='grey'):
  """Shows Plotly graph in Databricks"""
  pos = nx.spring_layout(graph)
  node_trace = get_nodes_from_graph(graph, pos, node_color)
  edge_trace, labels_trace = get_edges_from_graph(graph, pos, edge_color)
  fig = go.Figure(data=[edge_trace, node_trace, labels_trace],
               layout=go.Layout(
                  title='Company Ecosystem',
                  titlefont_size=16,                   
                  showlegend=False,
                  width=1600,
                  height=1000,
                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                  )
  fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers')) 
  fig.show()

# COMMAND ----------

import pandas as pd

def get_relations_df (results, col='relations'):
  """Shows a Dataframe with the relations extracted by Spark NLP"""
  rel_pairs=[]
  for rel in results[0][col]:
      rel_pairs.append((
        rel.result, 
        rel.metadata['entity1'], 
        rel.metadata['entity1_begin'],
        rel.metadata['entity1_end'],
        rel.metadata['chunk1'], 
        rel.metadata['entity2'],
        rel.metadata['entity2_begin'],
        rel.metadata['entity2_end'],
        rel.metadata['chunk2'], 
        rel.metadata['confidence']
    ))

  rel_df = pd.DataFrame(rel_pairs, columns=['relation','entity1','entity1_begin','entity1_end','chunk1','entity2','entity2_begin','entity2_end','chunk2', 'confidence'])

  return rel_df

# COMMAND ----------

# MAGIC %md
# MAGIC # Common Componennts
# MAGIC This pipeline will:
# MAGIC 1) Split Text into Sentences
# MAGIC 2) Split Sentences into Words
# MAGIC 3) Use Financial Text Embeddings, trained on SEC documents, to obtain numerical semantic representation of words
# MAGIC 
# MAGIC These components are common for all the pipelines we will use.

# COMMAND ----------

from pyspark.ml import Pipeline

# COMMAND ----------

def get_generic_base_pipeline():
  """Common components used in all pipelines"""
  document_assembler = nlp.DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

  sentence_detector = nlp.SentenceDetector()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")
  
  tokenizer = nlp.Tokenizer()\
      .setInputCols(["sentence"])\
      .setOutputCol("token")

  embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

  base_pipeline = Pipeline(stages=[
      document_assembler,
      sentence_detector,
      tokenizer,
      embeddings
  ])

  return base_pipeline
    
generic_base_pipeline = get_generic_base_pipeline()

# COMMAND ----------

# Text Classifier
def get_text_classification_pipeline(model):
  """This pipeline allows you to use different classification models to understand if an input text is of a specific class or is something else.
  It will be used to check where the first summary page of SEC10K is, where the sections of Acquisitions and Subsidiaries are, or where in the document
  the management roles and experiences are mentioned"""
  documentAssembler = nlp.DocumentAssembler() \
       .setInputCol("text") \
       .setOutputCol("document")

  useEmbeddings = nlp.UniversalSentenceEncoder.pretrained() \
      .setInputCols("document") \
      .setOutputCol("sentence_embeddings")

  docClassifier = nlp.ClassifierDLModel.pretrained(model, "en", "finance/models")\
      .setInputCols(["sentence_embeddings"])\
      .setOutputCol("category")

  nlpPipeline = Pipeline(stages=[
      documentAssembler, 
      useEmbeddings,
      docClassifier])
  
  return nlpPipeline

# COMMAND ----------

# MAGIC %md
# MAGIC # Sample Texts from Cadence Design System
# MAGIC Examples taken from publicly available information about Cadence in SEC's Edgar database [here](https://www.sec.gov/Archives/edgar/data/813672/000081367222000012/cdns-20220101.htm) and [Wikipedia](https://en.wikipedia.org/wiki/Cadence_Design_Systems)

# COMMAND ----------

!wget https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings_JSL/Finance/data/cdns-20220101.html.txt

# COMMAND ----------

with open('cdns-20220101.html.txt', 'r') as f:
  cadence_sec10k = f.read()
print(cadence_sec10k[:100])

# COMMAND ----------

pages = [x for x in cadence_sec10k.split("Table of Contents") if x.strip() != '']
print(pages[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Text Classification to find Relevant Parts of the Document: 10K Summary
# MAGIC In this case, we know page 0 is always the page with summary information about the company. However, let's suppose we don't know it. We can use Page Classification.
# MAGIC 
# MAGIC To check the SEC 10K Summary page, we have a specific model called `"finclf_form_10k_summary_item"`

# COMMAND ----------

classification_pipeline = get_text_classification_pipeline('finclf_form_10k_summary_item')
df = spark.createDataFrame([[pages[0]]]).toDF("text")
model = classification_pipeline.fit(df)
result = model.transform(df)

# COMMAND ----------

result.select('category.result').show()

# COMMAND ----------

# MAGIC %md
# MAGIC Confirmed, page 0 is where the 10K summary is!

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

summary_pipeline = Pipeline(stages=[
    generic_base_pipeline,
    ner_model_sec10k,
    ner_converter_sec10k
])

# COMMAND ----------

# MAGIC %md
# MAGIC Let's visualize the entities with Spark NLP Visualizer

# COMMAND ----------

from sparknlp_display import NerVisualizer
from sparknlp.base import LightPipeline

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

# MAGIC %md
# MAGIC We extract the organization (entity 'ORG' in the NER results)

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

# MAGIC %md
# MAGIC ## Normalizing the company name to query John Snow Labs datasources for more information about Cadence

# COMMAND ----------

# MAGIC %md
# MAGIC Sometimes, companies in texts use a non-official, abbreviated name. For example, we can find `Cadence`, `Cadence Inc`, `Cadence, Inc`, or many other variations, where the official name of the company os `CADENCE DESIGN SYSTEMS INC`, as per registered in SEC Edgar.

# COMMAND ----------

# MAGIC %md
# MAGIC Normalizing a company name is super important for data quality purposes. It will help us:
# MAGIC - Standardize the data, improving the quality;
# MAGIC - Carry out additional verifications;
# MAGIC - Join different databases or extract for external sources;

# COMMAND ----------

documentAssembler = nlp.DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")

embeddings = nlp.UniversalSentenceEncoder.pretrained("tfhub_use", "en") \
      .setInputCols("ner_chunk") \
      .setOutputCol("sentence_embeddings")
    
resolver = finance.SentenceEntityResolverModel.pretrained("finel_edgar_company_name", "en", "finance/models")\
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("normalization")\
      .setDistanceFunction("EUCLIDEAN")

pipelineModel = PipelineModel(
      stages = [
          documentAssembler,
          embeddings,
          resolver])

lp = LightPipeline(pipelineModel)

normalized_org = lp.fullAnnotate(ORG)
normalized_org

# COMMAND ----------

NORM_ORG = normalized_org[0]['normalization'][0].result
NORM_ORG

# COMMAND ----------

# MAGIC %md
# MAGIC ### NORMALIZED NAME
# MAGIC In Edgar, the company official is different! We need to take it before being able to augment with external information in EDGAR.
# MAGIC 
# MAGIC - Incorrect: `CADENCE DESIGN SYSTEMS, INC`
# MAGIC - Correct (Official): `CADENCE DESIGN SYSTEMS INC`

# COMMAND ----------

G.add_node(NORM_ORG, attr_dict={'entity': 'ORG'}),
G.add_edge(ORG, NORM_ORG, attr_dict={'relation': 'has_official_name'})  

# COMMAND ----------

# MAGIC %md
# MAGIC ## DATA AUGMENTATION WITH CHUNK MAPPER

# COMMAND ----------

# MAGIC %md
# MAGIC Once we have the normalized name of the company, we can use `John Snow Labs Chunk Mappers`. These are pretrained data sources, which are updated frequently and can be queried inside Spark NLP without sending any API call to any server.
# MAGIC 
# MAGIC In this case, we will use Edgar Database (`finmapper_edgar_companyname`)

# COMMAND ----------

documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

chunkAssembler = nlp.Doc2Chunk() \
    .setInputCols("document") \
    .setOutputCol("chunk") \
    .setIsArray(False)

CM = finance.ChunkMapperModel()\
      .pretrained("finmapper_edgar_companyname", "en", "finance/models")\
      .setInputCols(["chunk"])\
      .setOutputCol("mappings")
      
cm_pipeline = Pipeline(stages=[documentAssembler, chunkAssembler, CM])
fit_cm_pipeline = cm_pipeline.fit(empty_data)

df = spark.createDataFrame([[NORM_ORG]]).toDF("text")
r = fit_cm_pipeline.transform(df).collect()

# COMMAND ----------

# MAGIC %md
# MAGIC This is the information we got from that Chunk Mapper about Cadence.

# COMMAND ----------

mappings = r[0]['mappings']
for mapping in mappings:
  text = mapping.result
  relation = mapping.metadata['relation']
  print(f"{ORG} - has_{relation} - {text}")
    
  G.add_node(text, attr_dict={'entity': relation}),
  G.add_edge(ORG, text, attr_dict={'relation': 'has_' + relation.lower()})  

# COMMAND ----------

show_graph_in_plotly(G)

# COMMAND ----------

# MAGIC %md
# MAGIC ## NER and Relation Extraction
# MAGIC NER only extracts isolated entities by itself. But you can combine some NER with specific Relation Extraction Annotators trained for them, to retrieve if the entities are related to each other.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's suppose we want to extract information about Acquisitions and Subsidiaries. If we don't know where that information is in the document, we can again use or Text Classifiers to find it.

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
# MAGIC ### Acquisitions, Subsidiaries and Former Names
# MAGIC Let's use some NER models to obtain information about Organizations and Dates, and understand if:
# MAGIC - An ORG was acquired by another ORG
# MAGIC - An ORG is a subsidiary of another ORG
# MAGIC - An ORG name is an alias / abbreviation / acronym / etc of another ORG
# MAGIC 
# MAGIC We will use the deteceted `page[67]` as input

# COMMAND ----------

ner_model_date = finance.NerModel.pretrained("finner_sec_dates", "en", "finance/models")\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner_dates")

ner_converter_date = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner_dates"])\
        .setOutputCol("ner_chunk_date")

ner_model_org= finance.NerModel.pretrained("finner_orgs_prods_alias", "en", "finance/models")\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner_orgs")

ner_converter_org = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner_orgs"])\
        .setOutputCol("ner_chunk_org")\
        .setWhiteList(['ORG', 'PRODUCT', 'ALIAS'])

chunk_merger = finance.ChunkMergeApproach()\
        .setInputCols('ner_chunk_org', "ner_chunk_date")\
        .setOutputCol('ner_chunk')

pos = PerceptronModel.pretrained()\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("pos")

dependency_parser = DependencyParserModel().pretrained("dependency_conllu", "en")\
    .setInputCols(["sentence", "pos", "token"])\
    .setOutputCol("dependencies")

re_filter = finance.RENerChunksFilter()\
    .setInputCols(["ner_chunk", "dependencies"])\
    .setOutputCol("re_ner_chunk")\
    .setRelationPairs(["ORG-ORG", "ORG-DATE", "ORG-ROLE", "ROLE-DATE"])\
    .setMaxSyntacticDistance(10)

re_filter_alias = finance.RENerChunksFilter()\
    .setInputCols(["ner_chunk", "dependencies"])\
    .setOutputCol("re_ner_chunk_alias")\
    .setRelationPairs(["ORG-ALIAS"])\
    .setMaxSyntacticDistance(5)

reDL = finance.RelationExtractionDLModel().pretrained('finre_acquisitions_subsidiaries_md', 'en', 'finance/models')\
    .setInputCols(["re_ner_chunk", "sentence"])\
    .setOutputCol("relations_acq")\
    .setPredictionThreshold(0.1)

reDL_alias = finance.RelationExtractionDLModel()\
    .pretrained("finre_org_prod_alias", "en", "finance/models")\
    .setPredictionThreshold(0.8)\
    .setInputCols(["re_ner_chunk_alias", "sentence"])\
    .setOutputCol("relations_alias")

annotation_merger = finance.AnnotationMerger()\
    .setInputCols("relations_acq", "relations_alias")\
    .setOutputCol("relations")

nlpPipeline = Pipeline(stages=[
        generic_base_pipeline,
        ner_model_date,
        ner_converter_date,
        ner_model_org,
        ner_converter_org,
        chunk_merger,
        pos,
        dependency_parser,
        re_filter,
        re_filter_alias,
        reDL,
        reDL_alias,
        annotation_merger])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

light_model = LightPipeline(model)

# COMMAND ----------

sample_text = pages[67].replace("“", "\"").replace("”", "\"")

# COMMAND ----------

result = light_model.fullAnnotate(sample_text)
rel_df = get_relations_df(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Results

# COMMAND ----------

from sparknlp_display import RelationExtractionVisualizer

re_vis = viz.RelationExtractionVisualizer()
displayHTML(re_vis.display(result = result[0], relation_col = "relations", document_col = "document", exclude_relations = ["other", "no_rel"], return_html=True, show_relations=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inserting Nodes (Tags) and Relations into a Graph
# MAGIC Now, with entities and Relations connecting them, we can start populating the Graph of the company.

# COMMAND ----------

for t in rel_df.itertuples():
  relation = t.relation
  
  if relation in ['other', 'no_rel']:
    continue
  
  entity1 = t.entity1
  chunk1 = t.chunk1
  entity2 = t.entity2
  chunk2 = t.chunk2
  G.add_node(chunk1,  attr_dict={'entity': entity1})
  G.add_node(chunk2,  attr_dict={'entity': entity2})
  
  G.add_edge(ORG, chunk1, attr_dict={'relation': 'mentions_' + entity1.lower()})  
  G.add_edge(ORG, chunk2, attr_dict={'relation': 'mentions_' + entity2.lower()})  
  
  G.add_edge(chunk1, chunk2, attr_dict={'relation': relation.lower()})  
  

# COMMAND ----------

show_graph_in_plotly(G)

# COMMAND ----------

# MAGIC %md
# MAGIC ## People's Information
# MAGIC Let's also extract People's name with their current roles and past experiences in other companies (including the dates).

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
# MAGIC However, the model returned `other`. Why?

# COMMAND ----------

pages[4]

# COMMAND ----------

# MAGIC %md
# MAGIC Exploring the page we understand there is a lot of texts about something else which got into the same page. Sometimes, going into a smaller detail may be necessary.
# MAGIC 
# MAGIC Let's see what happens if we get `paragraphs` instead of `pages.`

# COMMAND ----------

paragraphs = [x for x in pages[4].split('\n') if x.strip() != '']

# COMMAND ----------

paragraphs

# COMMAND ----------



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

ner_model_role = finance.NerModel.pretrained("finner_org_per_role_date", "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner_role")

ner_converter_role = nlp.NerConverter()\
    .setInputCols(["sentence","token","ner_role"])\
    .setOutputCol("ner_chunk_role")

pos = PerceptronModel.pretrained()\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("pos")

dependency_parser = DependencyParserModel().pretrained("dependency_conllu", "en")\
    .setInputCols(["sentence", "pos", "token"])\
    .setOutputCol("dependencies")

re_ner_chunk_filter_role = finance.RENerChunksFilter()\
    .setInputCols(["ner_chunk_role", "dependencies"])\
    .setOutputCol("re_ner_chunk_role")\
    .setRelationPairs(["PERSON-ROLE", "ORG-ROLE", "DATE-ROLE"])

re_model_exp = finance.RelationExtractionDLModel.pretrained("finre_work_experience_md", "en", "finance/models")\
    .setInputCols(["re_ner_chunk_role", "sentence"])\
    .setOutputCol("relations")

nlpPipeline = Pipeline(stages=[
    generic_base_pipeline,
    ner_model_role,
    ner_converter_role,
    pos,
    dependency_parser,
    re_ner_chunk_filter_role,
    re_model_exp,
])


model = nlpPipeline.fit(empty_data)
light_model = LightPipeline(model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Results

# COMMAND ----------

sample_text = candidates[9]
sample_text

# COMMAND ----------

result = light_model.fullAnnotate(sample_text)
rel_df = get_relations_df(result)
rel_df[rel_df["relation"] != "other"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Results

# COMMAND ----------

from sparknlp_display import RelationExtractionVisualizer

re_vis = viz.RelationExtractionVisualizer()
displayHTML(re_vis.display(result = result[0], relation_col = "relations", document_col = "document", exclude_relations = ["other"], return_html=True, show_relations=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Adding to graph

# COMMAND ----------

for t in rel_df.itertuples():
  relation = t.relation
  if relation == 'other':
    continue
  entity1 = t.entity1
  chunk1 = t.chunk1
  entity2 = t.entity2
  chunk2 = t.chunk2
  G.add_node(chunk1,  attr_dict={'entity': entity1})
  G.add_node(chunk2,  attr_dict={'entity': entity2})
  
  G.add_edge(ORG, chunk1, attr_dict={'relation': 'mentions_' + entity1.lower()})  
  G.add_edge(ORG, chunk2, attr_dict={'relation': 'mentions_' + entity2.lower()})  
  
  G.add_edge(chunk1, chunk2, attr_dict={'relation': relation.lower()})  
  

# COMMAND ----------

show_graph_in_plotly(G)

# COMMAND ----------

# MAGIC %md
# MAGIC # Understanding the context of mentioned companies to identify COMPETITORS
# MAGIC Many Companies may be mentioned in the report. Most of them are just organizations in the ecosystem of the Cadence. Others, may be competitors.
# MAGIC 
# MAGIC We can analyze the surrounding context of the extracted `ORG` to check if they are competitors or not.

# COMMAND ----------

ner = finance.NerModel.pretrained("finner_orgs_prods_alias", "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")\
    .setWhiteList(['ORG', 'PRODUCT'])

assertion = finance.AssertionDLModel.pretrained("finassertion_competitors", "en", "finance/models")\
    .setInputCols(["sentence", "ner_chunk", "embeddings"])\
    .setOutputCol("assertion")

nlpPipeline = Pipeline(stages=[
    generic_base_pipeline,
    ner,
    ner_converter,
    assertion
])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

light_model = LightPipeline(model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get Results

# COMMAND ----------

sample_text = ["""In the rapidly evolving market, certain elements of our application compete with Microsoft, Google, InFocus, Bluescape, Mersive, Barco, Nureva and Prysm. But, Oracle  and IBM are out of our league."""]

chunks=[]
entities=[]
status=[]


light_result = light_model.fullAnnotate(sample_text)[0]

for n,m in zip(light_result['ner_chunk'],light_result['assertion']):
    chunks.append(n.result)
    entities.append(n.metadata['entity']) 
    status.append(m.result)

df = pd.DataFrame({'chunks':chunks, 'entities':entities, 'assertion':status})

# COMMAND ----------

df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Assertion Result

# COMMAND ----------

vis = viz.AssertionVisualizer()

vis.set_label_colors({'COMPETITOR':'#008080', 'NO_COMPETITOR':'#800080'})
    
light_result = light_model.fullAnnotate(sample_text)[0]

displayHTML(vis.display(light_result, 'ner_chunk', 'assertion', return_html=True))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Adding it to the graph

# COMMAND ----------

for t in df.itertuples():
  chunks = t.chunks
  entities = t.entities
  assertion = t.assertion

  G.add_node(chunks,  attr_dict={'entity': entities})
  
  G.add_edge(ORG, chunks, attr_dict={'relation': 'is_' + assertion.lower()})
  

# COMMAND ----------

show_graph_in_plotly(G)

# COMMAND ----------

# MAGIC %md
# MAGIC # Annex 1: Detecting Temporality and Certainty in Affirmations

# COMMAND ----------

ner_model_role = finance.NerModel.pretrained("finner_org_per_role_date", "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner_role")

ner_converter_role = nlp.NerConverter()\
    .setInputCols(["sentence","token","ner_role"])\
    .setOutputCol("ner_chunk")

assertion = finance.AssertionDLModel.pretrained("finassertion_time", "en", "finance/models")\
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")\
    .setMaxSentLen(1200)

assertion_pipeline = Pipeline(stages=[
    generic_base_pipeline,
    ner_model_role,
    ner_converter_role,
    assertion
])

empty_data = spark.createDataFrame([[""]]).toDF("text")

assertion_model = assertion_pipeline.fit(empty_data)

light_model_assertion = LightPipeline(assertion_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get Result

# COMMAND ----------

sample_text = ["""Joseph Costello was the CEO of the company since founded in 1988 until 1997. He was followed by Lip-Bu Tan for the 2009–2021 period. Currently, Anirudh Devgan is the CEO since 2021""",
              
              """In 2007, Cadence was rumored to be in talks with Kohlberg Kravis Roberts and Blackstone Group regarding a possible sale of the company.""",
              """In 2008, Cadence withdrew a $1.6 billion offer to purchase rival Mentor Graphics.""",
              
               """ The Cadence Giving Foundation will also support critical needs in areas such as diversity, equity and inclusion, environmental sustainability and STEM education.""",
              """This stand-alone, non-profit foundation will partner with other charitable initiatives to support critical needs in areas such as diversity, equity and inclusion, environmental sustainability and science, technology, engineering, and mathematics (“STEM”) education""",
              
              """Cadence employees could purchase common stock at a price equal to 85% of the lower of the fair market value at the beginning or the end of the applicable offering period"""]

chunks=[]               
entities=[]
status=[]

light_results = light_model_assertion.fullAnnotate(sample_text)

for light_result in light_results:
  for n,m in zip(light_result['ner_chunk'], light_result['assertion']):
      chunks.append(n.result)
      entities.append(n.metadata['entity']) 
      status.append(m.result)

df = pd.DataFrame({'chunks':chunks, 'entities':entities, 'assertion':status})

# COMMAND ----------

df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Assertion Result

# COMMAND ----------

vis = viz.AssertionVisualizer()

for light_result in light_results:
  displayHTML(vis.display(light_result, 'ner_chunk', 'assertion', return_html=True))


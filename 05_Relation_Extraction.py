# Databricks notebook source
# MAGIC %md You may find this series of notebooks at https://github.com/databricks-industry-solutions/jsl-financial-nlp

# COMMAND ----------

# MAGIC %pip install johnsnowlabs==4.2.3 networkx==2.5 decorator==5.0.9 plotly==5.1.0 

# COMMAND ----------

# MAGIC %md
# MAGIC ## NER and Relation Extraction
# MAGIC NER only extracts isolated entities by itself. But you can combine some NER with specific Relation Extraction Annotators trained for them, to retrieve if the entities are related to each other.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's suppose we want to extract information about Acquisitions and Subsidiaries. If we don't know where that information is in the document, we can again use or Text Classifiers to find it.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's resume the Graph creation, loading it from disk from previous step

# COMMAND ----------

from johnsnowlabs import nlp, finance, viz
import pickle

# COMMAND ----------

# MAGIC %run "./aux_visualization_functions"

# COMMAND ----------

# MAGIC %run "./aux_pipeline_functions"

# COMMAND ----------

generic_base_pipeline = get_generic_base_pipeline()

# COMMAND ----------

# load graph object from file
G = pickle.load(open('cadence.pickle', 'rb'))

# COMMAND ----------

with open('cadence_pages.pickle', 'rb') as f:
  pages = pickle.load(f)

# COMMAND ----------

print(pages[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Acquisitions, Subsidiaries and Former Names
# MAGIC Let's use some NER models to obtain information about Organizations and Dates, and understand if:
# MAGIC - An ORG was acquired by another ORG
# MAGIC - An ORG is a subsidiary of another ORG
# MAGIC - An ORG name is an alias / abbreviation / acronym / etc of another ORG
# MAGIC 
# MAGIC We will use the detected `page[67]` as input

# COMMAND ----------

from johnsnowlabs.nlp import LightPipeline

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

pos = nlp.PerceptronModel.pretrained()\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("pos")

dependency_parser = nlp.DependencyParserModel().pretrained("dependency_conllu", "en")\
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

nlpPipeline = nlp.Pipeline(stages=[
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

# COMMAND ----------

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

light_model = LightPipeline(model)

# COMMAND ----------

# We normalize some quotes
sample_text = pages[67].replace("“", "\"").replace("”", "\"")

# COMMAND ----------

result = light_model.fullAnnotate(sample_text)
rel_df = get_relations_df(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Results

# COMMAND ----------

re_vis = viz.RelationExtractionVisualizer()
displayHTML(re_vis.display(result = result[0], relation_col = "relations", document_col = "document", exclude_relations = ["other", "no_rel"], return_html=True, show_relations=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inserting Nodes (Tags) and Relations into a Graph
# MAGIC Now, with entities and Relations connecting them, we can start populating the Graph of the company.

# COMMAND ----------

ORG = [x for x in G.nodes()][0]
ORG

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

import pickle

paragraphs = pickle.load(open('cadence_people_paragraphs.pickle', 'rb'))

# COMMAND ----------

ner_model_role = finance.NerModel.pretrained("finner_org_per_role_date", "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner_role")

ner_converter_role = nlp.NerConverter()\
    .setInputCols(["sentence","token","ner_role"])\
    .setOutputCol("ner_chunk_role")

re_ner_chunk_filter_role = finance.RENerChunksFilter()\
    .setInputCols(["ner_chunk_role", "dependencies"])\
    .setOutputCol("re_ner_chunk_role")\
    .setRelationPairs(["PERSON-ROLE", "ORG-ROLE", "DATE-ROLE"])

re_model_exp = finance.RelationExtractionDLModel.pretrained("finre_work_experience_md", "en", "finance/models")\
    .setInputCols(["re_ner_chunk_role", "sentence"])\
    .setOutputCol("relations")

nlpPipeline = nlp.Pipeline(stages=[
    generic_base_pipeline,
    ner_model_role,
    ner_converter_role,
    pos,
    dependency_parser,
    re_ner_chunk_filter_role,
    re_model_exp,
])

# COMMAND ----------

model = nlpPipeline.fit(empty_data)
light_model = LightPipeline(model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Results

# COMMAND ----------

sample_text = paragraphs[9]
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
# MAGIC Saving the graph

# COMMAND ----------

import pickle

# save graph object to file
pickle.dump(G, open('cadence.pickle', 'wb'))

# COMMAND ----------

# MAGIC %md
# MAGIC # Now you are ready to go to next notebook, 06 Understanding Entities in Context

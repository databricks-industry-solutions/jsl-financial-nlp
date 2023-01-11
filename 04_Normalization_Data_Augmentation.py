# Databricks notebook source
# MAGIC %md You may find this series of notebooks at https://github.com/databricks-industry-solutions/jsl-financial-nlp

# COMMAND ----------

# MAGIC %md 
# MAGIC # Preinstallation
# MAGIC `johnsnowlabs` should come installed in your cluster. Just in case it is not, we install it.
# MAGIC We also install visualization libraries for rendering the graph.

# COMMAND ----------

# MAGIC %pip install networkx==2.5 decorator==5.0.9 plotly==5.1.0 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normalizing the company name to query John Snow Labs datasources for more information about Cadence

# COMMAND ----------

# MAGIC %md
# MAGIC Normalizing a company name is super important for data quality purposes. It will help us:
# MAGIC - Standardize the data, improving the quality;
# MAGIC - Carry out additional verifications;
# MAGIC - Join different databases or extract for external sources;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's resume the G creation, loading it from disk from previous step

# COMMAND ----------

from johnsnowlabs import nlp, finance, viz
import pickle

# COMMAND ----------

# MAGIC %run "./aux_visualization_functions"

# COMMAND ----------

# load graph object from file
G = pickle.load(open('/databricks/driver/cadence.pickle', 'rb'))

# COMMAND ----------

# MAGIC %md
# MAGIC Sometimes, companies in texts use a non-official, abbreviated name. For example, we can find `Cadence`, `Cadence Inc`, `Cadence, Inc`, or many other variations, where the official name of the company os `CADENCE DESIGN SYSTEMS INC`, as per registered in SEC Edgar.

# COMMAND ----------

# MAGIC %md
# MAGIC # Entity Resolution
# MAGIC To normalize names or map permutations or variations of strings to unique names or codes, we use Financial NLP `EntityResolvers`

# COMMAND ----------

from johnsnowlabs.nlp import LightPipeline

document_assembler = nlp.DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")

embeddings = nlp.UniversalSentenceEncoder.pretrained("tfhub_use", "en") \
      .setInputCols("ner_chunk") \
      .setOutputCol("sentence_embeddings")
    
resolver = finance.SentenceEntityResolverModel.pretrained("finel_edgar_company_name", "en", "finance/models")\
      .setInputCols(["sentence_embeddings"]) \
      .setOutputCol("normalization")\
      .setDistanceFunction("EUCLIDEAN")

pipeline = nlp.Pipeline(
      stages = [
          document_assembler,
          embeddings,
          resolver])

# COMMAND ----------

# MAGIC %md
# MAGIC Our unnormalized company name was our first node

# COMMAND ----------

ORG = [n for n in G.nodes()][0]
ORG

# COMMAND ----------

# MAGIC %md
# MAGIC Let's see it's official name

# COMMAND ----------

empty_data = spark.createDataFrame([[""]]).toDF("text")
pipelineModel = pipeline.fit(empty_data)

lp = LightPipeline(pipelineModel)

normalized_org = lp.fullAnnotate(ORG)
normalized_org

# COMMAND ----------

NORM_ORG = normalized_org[0]['normalization'][0].result
NORM_ORG

# COMMAND ----------

# MAGIC %md
# MAGIC Ok, it turns out it's `CADENCE DESIGN SYSTEMS INC`. We got our first insight, using pretrained Spark NLP data sources, in this case, an `EntityResolver` for company names normalization.
# MAGIC 
# MAGIC But Finance NLP has much more than that!

# COMMAND ----------

# MAGIC %md
# MAGIC ## DATA AUGMENTATION WITH CHUNK MAPPER

# COMMAND ----------

# MAGIC %md
# MAGIC Once we have the normalized name of the company, we can use `Finance NLP Chunk Mappers`. These are pretrained data sources, which are updated frequently and can be queried inside Spark NLP without sending any API call to any server.
# MAGIC 
# MAGIC In this case, we will use Edgar Database (`finmapper_edgar_companyname`)

# COMMAND ----------

documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

chunkAssembler = nlp.Doc2Chunk() \
    .setInputCols("document") \
    .setChunkCol("text") \
    .setOutputCol("chunk")

CM = finance.ChunkMapperModel()\
      .pretrained("finmapper_edgar_companyname", "en", "finance/models")\
      .setInputCols(["chunk"])\
      .setOutputCol("mappings")
      
cm_pipeline = nlp.Pipeline(stages=[documentAssembler, chunkAssembler, CM])
fit_cm_pipeline = cm_pipeline.fit(empty_data)

# COMMAND ----------

company_df = spark.createDataFrame([[NORM_ORG]]).toDF("text")

mappings = fit_cm_pipeline.transform(company_df)

# COMMAND ----------

mappings.show()

# COMMAND ----------

collected_mappings = mappings.select('mappings').collect()

for collected_mapping in collected_mappings:
  for relation in collected_mapping['mappings']:
      text = relation.result
      relation_name = relation.metadata['relation']
      print(f"{ORG} - has_{relation_name} - {text}")
      G.add_node(text, attr_dict={'entity': relation_name}),
      G.add_edge(ORG, text, attr_dict={'relation': 'has_' + relation_name.lower()})

# COMMAND ----------

show_graph_in_plotly(G)

# COMMAND ----------

import pickle

# save graph object to file
pickle.dump(G, open('/databricks/driver/cadence.pickle', 'wb'))

# Databricks notebook source
# MAGIC %md
# MAGIC # Understanding the context of mentioned companies to identify COMPETITORS
# MAGIC Many Companies may be mentioned in the report. Most of them are just organizations in the ecosystem of the Cadence. Others, may be competitors.
# MAGIC 
# MAGIC We can analyze the surrounding context of the extracted `ORG` to check if they are competitors or not.

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

from johnsnowlabs.nlp import LightPipeline

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

nlpPipeline = nlp.Pipeline(stages=[
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

ORG = [x for x in G.nodes()][0]
ORG

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

import pickle

# save graph object to file
pickle.dump(G, open('cadence.pickle', 'wb'))

# COMMAND ----------

# MAGIC %md
# MAGIC # Additional: Detecting Temporality and Certainty in Affirmations

# COMMAND ----------

# MAGIC %md
# MAGIC As an additional, extra step, let's explore the temporality and certainty of some entities using, again, Assertion Status.

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

assertion_pipeline = nlp.Pipeline(stages=[
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

# MAGIC %md
# MAGIC ### Visualize Assertion Result

# COMMAND ----------

vis = viz.AssertionVisualizer()

for light_result in light_results:
  displayHTML(vis.display(light_result, 'ner_chunk', 'assertion', return_html=True))


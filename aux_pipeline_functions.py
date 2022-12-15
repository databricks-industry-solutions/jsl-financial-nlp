# Databricks notebook source
from johnsnowlabs import nlp
from pyspark.ml import Pipeline


def get_generic_base_pipeline():
    """Common components used in all pipelines"""
    document_assembler = nlp.DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    sentence_detector = nlp.SentenceDetector() \
        .setInputCols(["document"]) \
        .setOutputCol("sentence")

    tokenizer = nlp.Tokenizer() \
        .setInputCols(["sentence"]) \
        .setOutputCol("token")

    embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base", "en") \
        .setInputCols(["sentence", "token"]) \
        .setOutputCol("embeddings")

    base_pipeline = nlp.Pipeline(stages=[
        document_assembler,
        sentence_detector,
        tokenizer,
        embeddings
    ])

    return base_pipeline
  
# Text Classifier
def get_text_classification_pipeline(model):
  """This pipeline allows you to use different classification models to understand if an input text is of a specific class or is something else.
  It will be used to check where the first summary page of SEC10K is, where the sections of Acquisitions and Subsidiaries are, or where in the document
  the management roles and experiences are mentioned"""
  document_assembler = nlp.DocumentAssembler() \
       .setInputCol("text") \
       .setOutputCol("document")

  use_embeddings = nlp.UniversalSentenceEncoder.pretrained() \
      .setInputCols("document") \
      .setOutputCol("sentence_embeddings")

  doc_classifier = finance.ClassifierDLModel.pretrained(model, "en", "finance/models")\
      .setInputCols(["sentence_embeddings"])\
      .setOutputCol("category")

  nlp_pipeline = nlp.Pipeline(stages=[
      document_assembler, 
      use_embeddings,
      doc_classifier])
  
  return nlp_pipeline

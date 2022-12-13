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
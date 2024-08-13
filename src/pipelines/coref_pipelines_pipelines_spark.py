import sparknlp
import pyspark.pandas as ps
import sparknlp.annotator as sa
import sparknlp.base as sb
from pyspark.ml.feature import Normalizer, SQLTransformer


def get_discourse_pipeline_spark():
    """
    Gets the sparknlp parts necessary for processing the dataset for discourse.
    """
    documenter = (
        sb.DocumentAssembler()
            .setInputCol("docs")
            .setOutputCol("document")
    )

    sentencer = (
        sa.SentenceDetector()
            .setInputCols(["document"])
            .setOutputCol("sentences")
            .setCustomBounds(['<NEW_SENT>'])
            .setUseCustomBoundsOnly(True)
    )

    tokenizer = (
        sa.Tokenizer()
            .setInputCols(["sentences"])
            .setOutputCol("token")
    )

    discourse_classifier = (
        sa.AlbertForSequenceClassification
            .load("discourse-classifier-albert")
            .setInputCols(["sentences", 'token'])
            .setOutputCol("discourse_label")
    )

    sent_finisher = (
        sb.Finisher()
            .setInputCols(["sentences"])
    )

    explode_sent = (
        SQLTransformer()
            .setStatement("""
             SELECT wayback_url, POSEXPLODE(finished_sentences) AS (sent_idx, sentence)
             FROM __THIS__
        """)
    )

    documenter_sent = (
        sb.DocumentAssembler()
            .setInputCol("sentence")
            .setOutputCol("sentence")
    )

    tokenizer_sent = (
        sa.Tokenizer()
            .setInputCols(["sentence"])
            .setOutputCol("token")
    )

    discourse_processing_pipeline = sb.Pipeline(stages=[
        documenter,
        sentencer,
        sent_finisher,
        explode_sent,
        documenter_sent,
        tokenizer_sent,
        discourse_classifier,
        # embeddings_finisher,
        # tok_finisher
      ]
    )

    return discourse_processing_pipeline


if __name__ == "__main__":


    from datasets import Features, Dataset
    import sparknlp
    spark = sparknlp.start(gpu=True)
    dataset = Dataset.load_from_disk('tmp-dataset-sentence-processed')
    df = dataset.to_pandas()
    df_for_discourse = df[['wayback_url', 'docs']]
    sdf = spark.createDataFrame(df_for_discourse)

    discourse_pipeline = get_discourse_pipeline_spark()


    """
    import sparknlp.annotator as sa
    import sparknlp.base as sb

    document_assembler = sa.DocumentAssembler() \
        .setInputCol('text') \
        .setOutputCol('document')
    
    tokenizer = sa.Tokenizer() \
        .setInputCols(['document']) \
        .setOutputCol('token')

    discourse_classifier = (
        sa.AlbertForSequenceClassification
            .load("discourse-classifier-albert")
            .setInputCols(["sentence", 'token'])
            .setOutputCol("discourse_label")
    )

    pipeline = sb.Pipeline(stages=[
        document_assembler, 
        tokenizer,
        discourse_classifier    
    ])

    example = spark.createDataFrame([["My name is Clara and I live in Berkeley, California."], ['My name is Wolfgang and I live in Berlin.']]).toDF("text")
    result = pipeline.fit(example).transform(example)
    result.select('discourse_label.metadata').show()    
    """
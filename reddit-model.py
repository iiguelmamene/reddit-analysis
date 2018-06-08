
from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *


from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import CountVectorizerModel

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os

def main(context):
  
    # SAVED PARQUETS
    # comments is the comments-minimal.json
    # submissions is the submissions.json
    
    # Read from JSON
    comments = sqlContext.read.json("comments-minimal.json.bz2")
    comments.registerTempTable("commentsTable")
    submissions = sqlContext.read.json("submissions.json.bz2")
    submissions.registerTempTable("submissionsTable")

    # Write the Parquets
    #comments.write.parquet("comments.parquet")
    #submissions.write.parquet("submissions.parquet")

    # Read the parquets
    #comments = sqlContext.read.parquet("comments.parquet")
    #comments.registerTempTable("commentsTable")
    #submissions = sqlContext.read.parquet("submissions.parquet")
    #submissions.registerTempTable("submissionsTable")

    # Read the CSV
    labels = sqlContext.read.format('csv').options(header='true', inferSchema='true').load("labeled_data.csv")
    labels.registerTempTable("labelsTable")

    dfTask2 = sqlContext.sql("SELECT commentsTable.* FROM commentsTable INNER JOIN labelsTable ON commentsTable.id = labelsTable.Input_id")

    # unigrams, bigrams, trigrams
    def unigrams_bigrams_trigrams(text):
        return parse-text.sanitize(text)

    udf_func = udf(unigrams_bigrams_trigrams, ArrayType(StringType()))
    dfTask4 = dfTask2.withColumn("udf_results", udf_func(col("body")))

    # countVectorizer
    if(not os.path.exists("cvModel")):
        cv = CountVectorizer(inputCol="udf_results", outputCol="features", binary=True, minDF=5.0)
        model = cv.fit(dfTask4)
        model.write().overwrite().save("cvModel")

    model = CountVectorizerModel.load("cvModel")
    dfTask6A = model.transform(dfTask4)
    dfTask6A.registerTempTable("dfTask6ATable")
    dfTask6B = sqlContext.sql("SELECT dfTask6ATable.*, IF(labelsTable.labeldjt=1, 1, 0) AS pos_label, if(labelsTable.labeldjt=-1, 1, 0) AS neg_label FROM dfTask6ATable INNER JOIN labelsTable ON dfTask6ATable.id = labelsTable.Input_id")
    dfTask6B.registerTempTable("dfTask6BTable")

    pos = sqlContext.sql('select pos_label as label, features from dfTask6BTable')
    neg = sqlContext.sql('select neg_label as label, features from dfTask6BTable')

    if(not os.path.exists("www/neg.model") or not os.path.exists("www/pos.model")):
        # Initialize two logistic regression models.
        poslr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10).setThreshold(0.2)
        neglr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10).setThreshold(0.25)
        # Binary classifier
        posEvaluator = BinaryClassificationEvaluator()
        negEvaluator = BinaryClassificationEvaluator()

        posParamGrid = ParamGridBuilder().addGrid(poslr.regParam, [1.0]).build()
        negParamGrid = ParamGridBuilder().addGrid(neglr.regParam, [1.0]).build()
        # 5 fold cross-validation pipeline.
        posCrossval = CrossValidator(
            estimator=poslr,
            evaluator=posEvaluator,
            estimatorParamMaps=posParamGrid,
            numFolds=2)
        negCrossval = CrossValidator(
            estimator=neglr,
            evaluator=negEvaluator,
            estimatorParamMaps=negParamGrid,
            numFolds=2)
        
        # Split the data 50/50
        posTrain, posTest = pos.randomSplit([0.5, 0.5])
        negTrain, negTest = neg.randomSplit([0.5, 0.5])
        # Train the models
        print("Training positive classifier...")
        posModel = posCrossval.fit(posTrain)
        print("Training negative classifier...")
        negModel = negCrossval.fit(negTrain)

        # Save the models and load them again later.
        posModel.write().overwrite().save("www/pos.model")
        negModel.write().overwrite().save("www/neg.model")

    # TO LOAD BACK IN
    posModel = CrossValidatorModel.load("www/pos.model")
    negModel = CrossValidatorModel.load("www/neg.model")

    
    dfTask8 = sqlContext.sql('SELECT commentsTable.id, commentsTable.body, commentsTable.created_utc, commentsTable.author_flair_text, submissionsTable.title, submissionsTable.pinned, commentsTable.score AS comment_score, submissionsTable.score AS story_score FROM commentsTable INNER JOIN submissionsTable ON RIGHT(commentsTable.link_id, 6)=submissionsTable.id')
    dfTask8 = dfTask8.sample(False, 0.05, None)

    # unigrams, bigrams, trigrams
    def unigrams_bigrams_trigrams(text):
        return parse-text.sanitize(text)

    udf_func = udf(unigrams_bigrams_trigrams, ArrayType(StringType()))
    dfTask9_1 = dfTask8.withColumn("udf_results", udf_func(col("body")))

    # countVectorizer
    model = CountVectorizerModel.load("cvModel")
    dfTask9_2 = model.transform(dfTask9_1)
    dfTask9_2.registerTempTable("dfTask9_2Table")

    
    dfTask9_3 = sqlContext.sql("SELECT * FROM dfTask9_2Table WHERE dfTask9_2Table.body NOT LIKE '%/s%' AND dfTask9_2Table.body NOT LIKE '&gt%'")
    dfTask9_3.registerTempTable("dfTask9_3Table")

    posResult_1 = posModel.transform(dfTask9_3)
    posResult_1.registerTempTable("posResult_1Table")
    posResult_2 = sqlContext.sql("SELECT posResult_1Table.id, posResult_1Table.body, posResult_1Table.author_flair_text, posResult_1Table.created_utc, posResult_1Table.title, posResult_1Table.comment_score, posResult_1Table.story_score, posResult_1Table.features, posResult_1Table.pinned, posResult_1Table.prediction AS pos FROM posResult_1Table")
    finalResult_1 = negModel.transform(posResult_2)
    finalResult_1.registerTempTable("finalResult_1Table")
    finalResult_2 = sqlContext.sql("SELECT finalResult_1Table.id, finalResult_1Table.body, finalResult_1Table.created_utc, finalResult_1Table.author_flair_text, finalResult_1Table.title, finalResult_1Table.comment_score, finalResult_1Table.story_score, finalResult_1Table.pos, finalResult_1Table.pinned, finalResult_1Table.prediction AS neg FROM finalResult_1Table")
    finalResult_2.registerTempTable("finalResult_2Table")

    if(not os.path.exists("final.parquet")):
        finalResult_2.write.parquet("final.parquet")

    final = sqlContext.read.parquet("final.parquet")
    final.registerTempTable("finalTable")

    # computations
    if(not os.path.exists("submissions.csv")):
        question1 = sqlContext.sql("SELECT (100 * sum(pos) / COUNT(*)) AS percent_pos, (100 * sum(neg) / COUNT(*)) AS percent_neg FROM finalTable")
        question1.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("submissions.csv")

    if(not os.path.exists("days.csv")):
        question2 = sqlContext.sql("SELECT DATE(from_unixtime(finalTable.created_utc)) AS date, 100*SUM(finalTable.pos)/COUNT(*) AS percent_pos, 100*SUM(finalTable.neg)/COUNT(*) AS percent_neg FROM finalTable GROUP BY date ORDER BY date")
        question2.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("days.csv")

    if(not os.path.exists("states.csv")):
        question3 = sqlContext.sql("SELECT finalTable.author_flair_text AS place, 100*SUM(finalTable.pos)/COUNT(*) AS percent_pos, 100*SUM(finalTable.neg)/COUNT(*) AS percent_neg FROM finalTable GROUP BY place ORDER BY place")
        question3.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("states.csv")

    if(not os.path.exists("comment.csv")):
        question4_comment = sqlContext.sql("SELECT finalTable.comment_score AS comment_score, 100*SUM(finalTable.pos)/COUNT(*) AS percent_pos, 100*SUM(finalTable.neg)/COUNT(*) AS percent_neg FROM finalTable GROUP BY comment_score ORDER BY comment_score")
        question4_comment.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("comment.csv")

    if(not os.path.exists("story.csv")):
        question4_story = sqlContext.sql("SELECT finalTable.story_score AS story_score, 100*SUM(finalTable.pos)/COUNT(*) AS percent_pos, 100*SUM(finalTable.neg)/COUNT(*) AS percent_neg FROM finalTable GROUP BY story_score ORDER BY story_score")
        question4_story.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("story.csv")

    if (not os.path.exists("pinned.csv")):
        EC_Pinned = sqlContext.sql("SELECT finalTable.pinned AS is_pinned, 100*SUM(finalTable.pos)/COUNT(*) AS percent_pos, 100*SUM(finalTable.neg)/COUNT(*) AS percent_neg FROM finalTable GROUP BY pinned ORDER BY pinned")
        EC_Pinned.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("pinned.csv")

if __name__ == "__main__":
    conf = SparkConf().setAppName("reddit-analysis")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("parse-text.py")
    import parse-text
    main(sqlContext)

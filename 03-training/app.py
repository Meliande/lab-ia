from datetime import datetime, date
import pandas as pd
import plotly.express as px

from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler

### Modelos utilizados ###

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier

from pyspark.sql import Row
from pyspark.sql import SQLContext

spark = SparkSession \
    .builder \
    .appName("ML Pipeline") \
    .getOrCreate()

sc = SparkContext.getOrCreate();

url = "data/preprocessado/Farol200/out.csv"
from pyspark import SparkFiles
sc.addFile(url)
sqlContext = SQLContext(sc)

Farol200 = sqlContext.read.csv(SparkFiles.get("out.csv"), header=True, inferSchema= True)
Farol200 = Farol200.drop("_c0")
Farol200.toPandas()


Farol200 = Farol200.drop('categoria')

Farol200 = Farol200.filter(Farol200.Categoria_Indexada < 2)

Farol200.toPandas()

informacoes_necessarias = ['Total de Chuvas(mm)','Média diária de chuvas(mm)']

assembler = VectorAssembler(inputCols=informacoes_necessarias, outputCol='informacoes')

dfFarol200 = assembler.transform(Farol200)

dfFarol200.toPandas()

(treinoFarol200, testeFarol200) = dfFarol200.randomSplit([0.8,0.2])

treinoFarol200.toPandas()

### Definindo os modelos ###

gbt = GBTClassifier(labelCol="Categoria_Indexada", featuresCol="informacoes", maxIter=10)
dt = DecisionTreeClassifier(labelCol='Categoria_Indexada',featuresCol='informacoes')
rf = RandomForestClassifier(labelCol='Categoria_Indexada',featuresCol='informacoes',maxDepth=5)

gbtModel = gbt.fit(treinoFarol200) ### Gradient Boosted Tree Classifier

rfModel = rf.fit(treinoFarol200) ### Random Forest Classifier

dtModel = dt.fit(treinoFarol200) ### Decision Tree classfier

rfModel.write().overwrite().save('data/models/rf')

gbtModel.write().overwrite().save('data/models/gbt')

dtModel.write().overwrite().save('data/models/dt')

### Testando os modelos ###

gbtPredicao = gbtModel.transform(testeFarol200)

gbtPredicao.toPandas()

rfPredicao = rfModel.transform(testeFarol200)

rfPredicao.toPandas()

dtPredicao = dtModel.transform(testeFarol200)

dtPredicao.toPandas()

### Avaliando os modelos ### 

### Definindo os avaliadores ###

### Documentação => https://spark.apache.org/docs/latest/mllib-evaluation-metrics.html ###

### https://towardsdatascience.com/the-f1-score-bec2bbc38aa6 ###

acuracia = MulticlassClassificationEvaluator(labelCol='Categoria_Indexada',predictionCol='prediction',metricName='accuracy')
f1 = MulticlassClassificationEvaluator(labelCol='Categoria_Indexada',predictionCol='prediction',metricName='f1')
precisaoPonderada = MulticlassClassificationEvaluator(labelCol='Categoria_Indexada',predictionCol='prediction',metricName='weightedPrecision')
weightedRecall = MulticlassClassificationEvaluator(labelCol='Categoria_Indexada',predictionCol='prediction',metricName='weightedRecall')

### Resultados da Acurácia ###

gbtAcuracia = acuracia.evaluate(gbtPredicao)
rfAcuracia = acuracia.evaluate(rfPredicao)
dtAcuracia = acuracia.evaluate(dtPredicao)
print('Acurácia do teste Árvore de Decisão (Gradiente Boosting) = ', gbtAcuracia)
print('Acurácia do teste Árvore Aleatória = ', rfAcuracia)
print('Acurácia do teste Árvore de Decisão = ', dtAcuracia)



### Resultados do F1 ###

gbtF1 = f1.evaluate(gbtPredicao)
rfF1 = f1.evaluate(rfPredicao)
dtF1 = f1.evaluate(dtPredicao)
print('F1 do teste Árvore de Decisão (Gradiente Boosting) = ', gbtF1)
print('F1 do teste Árvore Aleatória = ', rfF1)
print('F1 do teste Árvore de Decisão = ', dtF1)

### Resultados do Precisão Ponderada ###

gbtPP = precisaoPonderada.evaluate(gbtPredicao)
rfPP = precisaoPonderada.evaluate(rfPredicao)
dtPP = precisaoPonderada.evaluate(dtPredicao)
print('Precisão Ponderada do teste Árvore de Decisão (Gradiente Boosting) = ', gbtPP)
print('Precisão Ponderada do teste Árvore Aleatória = ', rfPP)
print('Precisão Ponderada do teste Árvore de Decisão = ', dtPP)

### Resultados do weightedRecall ###

### Recall é a razão entre o número de positivos verdadeiros (pv) e a soma dos positivos verdadeiros (pv) e falsos negativos (fn) => pv/(pv+fn)

gbtWR = weightedRecall.evaluate(gbtPredicao)
rfWR = weightedRecall.evaluate(rfPredicao)
dtWR = weightedRecall.evaluate(dtPredicao)
print('Recall do teste Árvore de Decisão (Gradiente Boosting) = ', gbtWR)
print('Recall do teste Árvore Aleatória = ', rfWR)
print('Recall do teste Árvore de Decisão = ', dtWR)
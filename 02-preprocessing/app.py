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

spark = SparkSession \
    .builder \
    .appName("ML Pipeline") \
    .getOrCreate()

df = spark.read.option("delimiter", ";").option("header", True).csv('Data/inmet_filtered_A401_H_2000-05-12_2023-05-16.csv')

dfTransformado = df.withColumn('PRECIPITACAO TOTAL, HORARIO(mm)', regexp_replace('PRECIPITACAO TOTAL, HORARIO(mm)', ',', '.').cast(DoubleType()))

dfTransformado = dfTransformado.drop(
           'PRESSAO ATMOSFERICA REDUZIDA NIVEL DO MAR, AUT(mB)',
           'PRESSAO ATMOSFERICA REDUZIDA NIVEL DO MAR, AUT(mB)',
           'PRESSAO ATMOSFERICA MAX.NA HORA ANT. (AUT)(mB)',
           'PRESSAO ATMOSFERICA MIN. NA HORA ANT. (AUT)(mB)',
           'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA(mB)',
           'TEMPERATURA DA CPU DA ESTACAO(°C)',
           'TEMPERATURA DO AR - BULBO SECO, HORARIA(°C)',
           'TEMPERATURA DO PONTO DE ORVALHO(°C)',
           'TEMPERATURA MAXIMA NA HORA ANT. (AUT)(°C)',
           'TEMPERATURA MINIMA NA HORA ANT. (AUT)(°C)',
           'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT)(°C)',
           'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT)(°C)',
           'TENSAO DA BATERIA DA ESTACAO(V)',
           'UMIDADE REL. MAX. NA HORA ANT. (AUT)(%)',
           'UMIDADE REL. MIN. NA HORA ANT. (AUT)(%)',
           'UMIDADE RELATIVA DO AR, HORARIA(%)',
           'VENTO, DIRECAO HORARIA (gr)(° (gr))',
           'VENTO, RAJADA MAXIMA(m/s)',
           'VENTO, VELOCIDADE HORARIA(m/s)',
           'Unnamed: 22',
           '_c22'
)

df = dfTransformado.withColumn("Inicio_Semana",date_sub(next_day(col("Data Medicao"),"sunday"),7))\
                    .groupBy("Inicio_Semana").agg\
                        (sum("PRECIPITACAO TOTAL, HORARIO(mm)").cast("float").alias("Total de Chuvas(mm)"),\
                         sum("PRECIPITACAO TOTAL, HORARIO(mm)").cast("float").alias("Média diária de chuvas(mm)"))\
                    .orderBy("Inicio_Semana")

df = df.select('Inicio_Semana','Total de Chuvas(mm)',col('Média diária de chuvas(mm)')/ 7 )

df = df.withColumnRenamed("(Média diária de chuvas(mm) / 7)","Média diária de chuvas(mm)")

dfCompleto = df.withColumn('Semana_Ano',weekofyear(df.Inicio_Semana))
dfCompleto.toPandas()

dfCompleto = dfCompleto.withColumn("Ano", substring(dfCompleto.Inicio_Semana, 1,4))

dfCompleto.toPandas()

inema = spark.read.option("header",True).csv("Data/inema_filtered_balneabilidade_farol_barra.csv")

inema = inema.withColumnRenamed("01/2007","numero_boletim").withColumnRenamed("Farol da Barra - SSA FB 100","ponto_codigo").withColumnRenamed("Indisponível","categoria")

inema = inema.withColumn("Ano",substring(inema.numero_boletim, 4,7)).withColumn("Semana_Ano", substring(inema.numero_boletim, 1,2))

inema.toPandas()

dfCompleto = dfCompleto.join(inema,["Ano","Semana_Ano"])

dfCompleto = dfCompleto.drop("numero_boletim","Ano","Semana_Ano")

dfCompleto.toPandas()

### Contabilizando os Nulls por coluna ###

dfCompleto.select([count(when(isnull(c), c)).alias(c) for c in dfCompleto.columns]).toPandas()

### Removendo os Nulls ####

dfCompleto = dfCompleto.replace('?', None).dropna(how='any')

### Transformando os valores qualitativos em numéricos => 0 = Própria | 1 = Imprópria | 2 = Indisponível ###

dfCompleto = StringIndexer(
    inputCol='categoria', 
    outputCol='Categoria_Indexada', 
    handleInvalid='keep').fit(dfCompleto).transform(dfCompleto)

    Farol200 = dfCompleto.filter(dfCompleto.ponto_codigo == "Farol da Barra - SSA FB 200")

    Farol200.show()

from pathlib import Path  

filepath = Path('data/preprocessado/Farol200/out.csv')  

filepath.parent.mkdir(parents=True, exist_ok=True)  

Farol200.toPandas().to_csv(filepath)  
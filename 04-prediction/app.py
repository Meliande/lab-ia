
# load Flask 
import flask
from flask import jsonify
import requests
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.types import StructType,StructField, StringType, IntegerType, FloatType
from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler



app = flask.Flask(__name__)

# define a predict function as an endpoint 
@app.route("/", methods=["GET","POST"])
def prediction():

    data = {}
    params = flask.request.json
    if params is None:
        params = flask.request.args


    # params = request.get_json()
    if "data" in params:
        spark = SparkSession \
        .builder \
        .appName("ML Pipeline") \
        .getOrCreate()

        sc = SparkContext.getOrCreate();

        rfModel = RandomForestClassificationModel.load("data/models/rf")
        data2 = [
            (float(params['data']['total_chuvas']),float(params['data']['media_diaria'])),
        ]

        schema = StructType([ \
            StructField("Total de Chuvas(mm)",FloatType(),True), \
            StructField("Média diária de chuvas(mm)",FloatType(),True), \
        ])
        
        df = spark.createDataFrame(data=data2,schema=schema)

        informacoes_necessarias = ['Total de Chuvas(mm)','Média diária de chuvas(mm)']
        assembler = VectorAssembler(inputCols=informacoes_necessarias, outputCol='informacoes')
        df = assembler.transform(df)

        teste = rfModel.transform(df)

        saida = (teste.first()['prediction'])

        if saida == 0.0 :
            data["previsao"] = "Própria"
        else:
            data["previsao"] = "Imrópria"
    return jsonify(data)
    
# start the flask app, allow remote connections
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=3000)

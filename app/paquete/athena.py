from ..data.preprocess import engine
import boto3
import awswrangler as wr
import pandas as pd
from datetime import datetime, timedelta

def load_data(df, table_name, folder, cols):
    boto3.setup_default_session(region_name='us-east-1', profile_name='bi-athena')
    # Storing data on Data Lake
    if df.shape[0] > 0:
        wr.s3.to_parquet(
            df=df,
            path=f"s3://overall-datalake-rw-dev/dataset/{table_name}", # folder = dataset?
            dataset=True,
            database="livetradebi",
            table=table_name,
            mode="overwrite",
            partitions_cols = cols
        )
        print ( format(datetime.today()-timedelta(hours=5),'%Y-%m-%d %H:%M:%S') + " Data insertada correctamente") 
    else:
        print ( format(datetime.today()-timedelta(hours=5),'%Y-%m-%d %H:%M:%S') + " No existe registros para insertar")
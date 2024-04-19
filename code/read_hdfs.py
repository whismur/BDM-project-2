import os

import pyarrow as pa
import pyarrow.parquet as pq
from run_pipeline import hdfs_env_setup

hdfs_env_setup()
filesystem = pa.fs.HadoopFileSystem(
    host=os.environ["HDFS_HOST"], port=int(os.environ["HDFS_PORT"])
)

parquet_file = "/user/bdm/P1/total.parquet"

table = pq.read_table(parquet_file, filesystem=filesystem)

df = table.to_pandas()

print(df.head())

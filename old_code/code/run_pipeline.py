import asyncio
import os

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from preprocess import preprocess


def hdfs_env_setup() -> None:
    """Set up HDFS environment variables (needs HADOOP_HOME, JAVA_HOME, CLASSPATH)"""

    load_dotenv()
    os.environ["CLASSPATH"] = (
        os.popen("$HADOOP_HOME/bin/hadoop classpath --glob").read().strip()
    )


async def main():
    use_hdfs = True
    parquet_file = "/user/bdm/P1/total.parquet"  # "data/out/total.parquet"

    df = await preprocess()

    # Set filesystem
    if use_hdfs:
        hdfs_env_setup()
        filesystem = pa.fs.HadoopFileSystem(
            host=os.environ["HDFS_HOST"], port=int(os.environ["HDFS_PORT"])
        )
    else:
        filesystem = pa.fs.LocalFileSystem()

    # Save parquet
    schema = pa.Schema.from_pandas(df)
    table = pa.Table.from_pandas(df, schema=schema)
    with pq.ParquetWriter(parquet_file, schema, filesystem=filesystem) as writer:
        writer.write_table(table)


if __name__ == "__main__":
    asyncio.run(main())

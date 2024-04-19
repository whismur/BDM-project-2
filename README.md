# BDM Project Part 1

## Structure

* `code/`: all code to run the pipeline.
    * `preprocess.py`: preprocessing funcionality.
    * `run_pipeline.py`: main code to run the pipeline can use local or HDFS.
    * `read_hdfs.py`: code to test reading the output from HDFS.
* `data/`: all the data nnedeed for the pipeline.
    * `BCN_UNITATS_ADM`: GeoSpatial Barcelona Neighborhoods.
    * `idealista`: housing.
    * `lookup_tables`
    * `opendatabcn-incidents`: incidents.
    * `opendatabcn-income`: income.
    * `out/total.parquet`: local output file for testing.

## Instructions

1. Install the required python packages in `requirements.txt`.
2. Copy `.env.sample` file to a `.env` file and set the env variables for your system.
3. Run the `python code/run_pipeline.py` script.

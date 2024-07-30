import argparse
import os
import pandas as pd
from ..utils.datasets import get_json_data, process_json_data



def main(args):
    """
    Main function for processing JSON data, training models, and saving outputs.

    Args:
        args (Any): Command-line arguments and options.

    Returns:
        None
    """
    from utils.datasets import COLS_DROPPED_PFCP, COLS_DROPPED_TSTAT, COLS_DROPPED_CIC

    # COMMENT IN AGAIN when infrastructure
    # Get the data and process them into a Pandas data frame
    #json_data = get_json_data(args.url, args.query, args.aggregator)
    #data = process_json_data(json_data, args.aggregator)

    if args.aggregator == "pfcpflowmeter":
        cols_dropped = COLS_DROPPED_PFCP

    elif args.aggregator == "tstat":
        cols_dropped = COLS_DROPPED_TSTAT

    elif args.aggregator == "cicflowmeter":
        cols_dropped = COLS_DROPPED_CIC

    path = args.aggregator + ".csv"
    data = pd.read_csv(os.path.join('../data', 'training', path))

    data = data.drop(cols_dropped, axis=1)
    data = data.sort_index(axis=1)

    print(data.head())
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receive data required for adversarial data generation.")
    parser.add_argument("--aggregator", type=str, default='tstat', help="Aggregator to use. Can be 'pfcpflowmeter', 'tstat', 'cicflowmeter'. ")
    parser.add_argument("--url", type=str, default="http://83.212.72.97:9200", help="API URL to request data.")
    parser.add_argument("--query", type=dict, default={"_source": ["*"]}, help="Query to retrieve from ElasticSearch")

    args = parser.parse_args()
    main(args)
from typing import Dict, Union
import pandas as pd
import requests
import logging
import json

COLS_DROPPED_PFCP = ["flow ID", " source IP", " source port", " destination IP", " destination port", " protocol"]

COLS_DROPPED_CIC = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp"]

COLS_DROPPED_TSTAT = ['s_ttl_min', 'c_pkts_all', 's_rtt_cnt', 'c_mss_min', 'c_tls_SNI', 's_win_max', 's_rst_cnt', 
                           'c_sack_cnt', 'c_appdataB', 's_fin_cnt', 's_win_min', 'c_mss_max', 'c_pkts_ooo', 's_cwin_min', 
                           'c_pkts_push', 's_npnalpn', 'c_pkts_reor', 'p2p_t', 's_pkts_rto', 's_pkts_unrto', 's_iscrypto',
                           'c_win_scl', 'c_mss', 'ed2k_chat', 'c_bytes_retx', 'res_tm', 'c_npnalpn', 'c_pkts_rto',
                           's_bytes_retx', 's_isint', 'c_pkts_retx', 's_mss', 'p2p_st', 's_pkts_ooo', 'ed2k_sig', 'c_cwin_min',
                            's_ip', 's_pkts_data', 'c_pkts_unrto', 'http_req_cnt', 'c_rtt_cnt', 'con_t', 'c_win_0', 'c_cwin_max',
                            'c_appdataT', 's_port', 's_last_handshakeT', 'http_res_cnt', 'c_pkts_fc', 'http_t', 'c_pkts_unfs', 's_cwin_ini', 'c_fin_cnt',
                            'c_bytes_all', 'dns_rslv', 's_ttl_max', 's_f1323_opt', 'c_pkts_fs', 'first', 'http_res', 'c_syn_retx', 's_tls_SCN',
                            'ed2k_c2c', 's_win_0', 'c_pkts_data', 's_appdataT', 'c_isint', 's_bytes_uniq', 's_tm_opt', 's_pkts_fs', 'c_ack_cnt', 
                            'c_last_handshakeT', 'ed2k_c2s', 's_pkts_retx', 's_appdataB', 's_pkts_fc', 'c_bytes_uniq', 'last', 'fqdn', 'c_pkts_dup', 
                            's_pkts_unk', 'c_ip', 's_sack_opt', 's_pkts_unfs', 'c_sack_opt', 'c_pkts_unk', 's_pkts_push', 'c_ack_cnt_p', 'c_syn_cnt', 
                            'c_iscrypto', 's_mss_min', 's_syn_cnt', 'c_ttl_min', 's_bytes_all', 's_pkts_dup', 'c_win_max', 'c_cwin_ini', 's_win_scl', 
                            'c_f1323_opt', 's_syn_retx', 'ed2k_data', 'c_tls_sesid', 'c_ttl_max', 'c_win_min', 's_pkts_reor', 'c_tm_opt', 's_sack_cnt', 
                            's_cwin_max', 's_mss_max', 'req_tm', 'c_rst_cnt']



def get_json_data(url: str, query: dict, aggregator: str, max_size: int = 10000) -> list:
    """
    Retrieves JSON data from the specified URL using a GET request.
    Args:
        url (str): The URL to retrieve JSON data from.
        query (dict): The query parameters to include in the request.
        aggregator (str): Aggregator to use for trainings.
        max_size (int, optional): The maximum number of records to retrieve. Default is 10000.
    Returns:
        list: A list of JSON records retrieved from the specified URL.
    """
    headers = {'Content-Type': 'application/json'}
    # Add max_size to the query parameters
    query['size'] = min(max_size, query.get('size', max_size))

    api_url = f'{url}/{aggregator}/_search?filter_path=hits.hits._source'
    try:
        response = requests.get(api_url, headers=headers, data=json.dumps(query))
        response.raise_for_status()  # Raise an exception for HTTP errors
        result = response.json()

        print("Data was loaded in json.")

        return result.get('hits', {}).get('hits', [])
    except requests.RequestException as e:
        logging.error(f"Error during request: {e}")
        raise e

def process_json_data(json_data: Dict[str, Union[str, dict]], aggregator: str) -> pd.DataFrame:
    """
    Processes JSON data and returns a Pandas DataFrame with selected columns removed.
    
    Args:
        json_data (dict): The JSON data to be processed.
        aggregator (str): The type of aggregator for which to process the data.
        
    Returns:
        pandas.DataFrame: A DataFrame containing the processed data.
    """
    data = pd.json_normalize(json_data)

    # Rename columns using the remove_prefix function
    data = data.rename(columns=lambda x: x.replace("_source.", "") if '_source.' in x else x)

    if aggregator == "pfcpflowmeter":
        cols_dropped = COLS_DROPPED_PFCP

    elif aggregator == "tstat":
        cols_dropped = COLS_DROPPED_TSTAT
        
    elif aggregator == "cicflowmeter":
        cols_dropped = COLS_DROPPED_CIC

    else:
        raise ValueError("Invalid aggregator type.")

    data = data.drop(cols_dropped, axis=1)
    data = data.sort_index(axis=1)
    print("Data was processed in csv.")
    return data
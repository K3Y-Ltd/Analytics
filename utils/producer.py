import time
import os
import csv
import json
import sys
from typing import Union
from kafka import KafkaProducer
import argparse


def connect_publisher_client(broker_ip_address: str, broker_port: str) -> KafkaProducer:
    """
        Creates a KafkaProducer instance and returns it.
    """
    broker = f'{broker_ip_address}:{broker_port}'
    producer = KafkaProducer(
        bootstrap_servers=broker
    )

    print("Trying to establish connection to brokers...")
    print("Connection status: ", producer.bootstrap_connected())

    # Validate if connection to brokers is ready
    if not producer.bootstrap_connected():
        sys.exit("Failed to connect to brokers.")

    return producer

def main(producer: KafkaProducer, topic: [str], file_path: str, interval: Union[float, int]) -> None:
    """
        This function starts reading the CSV file and sends each line to the Kafka topic.
    """
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        # Get the column names from the first row
        fieldnames = csv_reader.fieldnames
        print("Fieldnames: ", fieldnames)

        for row in csv_reader:
            # Convert each row to a JSON object using the column names as fields
            json_data = {field: row[field] for field in fieldnames if field!=''}
            payload = json.dumps(json_data, indent=4)
            print("Payload: ", payload)
            producer.send(
                topic=topic,
                value=payload.encode('utf-8'),
                timestamp_ms=time.time_ns() // 1000000
            )
            producer.flush()
            time.sleep(interval)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train and validate linear models with k-fold cross-validation.")
    args.add_argument("--broker_ip_address", type=str, default='83.212.72.97', help="....")
    args.add_argument("--broker_port", type=str, default="9094", help="....")
    args.add_argument("--topic", type=str, default="pfcpflowmeter", help="One of 'cicflowmeter', 'tstat', 'pfcpflowmeter'.")
    args.add_argument("--interval", type=int, default=0.1, help="How often payloads are served.")
    args = args.parse_args()

    # csv_file_path = '../data/training/tstat.csv'
    # csv_file_path = '../data/training/cicflowmeter.csv'
    csv_file_path = '../data/training/pfcpflowmeter.csv'

    if not os.path.exists(csv_file_path):
        sys.exit(f"File not found: {csv_file_path}")

    producer = connect_publisher_client(args.broker_ip_address, args.broker_port)

    main(producer, topic=args.topic, file_path=csv_file_path, interval=args.interval)


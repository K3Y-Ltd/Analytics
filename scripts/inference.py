import os
import argparse
import time
import sys

from confluent_kafka import Consumer, KafkaError
import pandas as pd
import json
from kafka import KafkaProducer
from utils.datasets import COLS_DROPPED_PFCP, COLS_DROPPED_CIC, COLS_DROPPED_TSTAT
from utils.utils import download_file





# Function to consume messages from Kafka topic
def main(producer, consumer, consumer_topics, producer_topics, url, save_path, model_name):

    # Load the model and start the consumer while the publisher is running
    loaded_model = download_file(url, save_path, model_name)
    print(f"Best parameters:  {loaded_model}")
    consumer.subscribe([consumer_topics])

    try:
        while True:
            msg = consumer.poll(1.0)  # Adjust the timeout as needed

            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(msg.error())
                    break

            # Preprocess the received message to pd format
            instance = msg.value().decode('utf-8')
            instance = pd.DataFrame([json.loads(instance)])

            # Select columns to drop according to format
            if 'cicflowmeter' in consumer_topics:
                cols_dropped = COLS_DROPPED_CIC
            elif 'tstat' in consumer_topics:
                cols_dropped = COLS_DROPPED_TSTAT
            elif 'pfcpflowmeter' in consumer_topics:
                cols_dropped = COLS_DROPPED_PFCP
            else:
                sys.exit("Consumer not available.")

            # Process the instance
            instance = instance.drop(cols_dropped, axis=1)
            instance = instance.drop('Label', axis=1)
            instance = instance.sort_index(axis=1)

            # Make a prediction using the machine learning model
            prediction = loaded_model.predict(instance).item()

            # Add the prediction to the instance row and send it to the kafka topick
            instance["Label"] = prediction
            payload = json.dumps(instance.squeeze().to_dict(), indent=4)
            producer.send(
                topic=producer_topics,
                value=payload.encode('utf-8'),
                timestamp_ms=time.time_ns() // 1000000
            )
            producer.flush()


    except:
        producer.close()
        consumer.close()




if __name__ == "__main__":

    args = argparse.ArgumentParser(description="Train and validate linear models with k-fold cross-validation.")
    args.add_argument("--url", type=str, default='http://83.212.72.97:5000/analytics/', help="API URL to send models and metadata.")
    args.add_argument("--save_path", type=str, default="./downloaded_models", help="Where to store the models downloaded from the ElasticSearch storage service.")
    args.add_argument("--model", type=str, default="SVM.joblib", help="Name for the model(s) downloaded from the ElasticSearch storage service.")
    args.add_argument("--broker_ip_address", type=str, default='83.212.72.97', help="....")
    args.add_argument("--broker_port", type=str, default="9094", help="....")
    args.add_argument("--group_id", type=str, default="development", help="....")
    args.add_argument("--consumer_topics", type=str, default="pfcpflowmeter", help="One of 'cicflowmeter', 'tstat', 'pfcpflowmeter'.")
    args.add_argument("--producer_topics", type=str, default="pfcpflowmeter-inference", help="One of 'tstat-inference', 'cicflowmeter-inference', 'pfcpflowmeter-inference'.")

    args = args.parse_args()



    bootstrap_servers = f'{args.broker_ip_address}:{args.broker_port}'
    producer_config = {
        'bootstrap_servers': bootstrap_servers,
    }

    # Create a Kafka consumer
    consumer_config = {
        'bootstrap.servers': bootstrap_servers,
        'group.id': args.group_id,
        'auto.offset.reset': 'latest'
    }

    producer = KafkaProducer(**producer_config)
    consumer = Consumer(consumer_config)


    # Start consuming messages
    url = os.path.join(args.url, args.consumer_topics)
    save_path = os.path.join(args.save_path, args.consumer_topics)
    main(producer, consumer, args.consumer_topics, args.producer_topics, url, save_path, args.model)

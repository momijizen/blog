---
title: "Streaming Data Using Kafka"
date: 2021-09-05
categories:
  - Blog
tags:
  - Kafka
  - IBM Lab
link:

---

# 1 - Download and extract Kafka
Open a new terminal.
Download Kafka, by running the command below:
```
wget https://archive.apache.org/dist/kafka/2.8.0/kafka_2.12-2.8.0.tgz

```

Extract kafka from the zip file by running the command below.
```
tar -xzf kafka_2.12-2.8.0.tgz
```
This creates a new directory 'kafka_2.12-2.8.0' in the current directory.

# 2 - start ZooKeeper
ZooKeeper is required for Kafka to work. Start the ZooKeeper server.
```
cd kafka_2.12-2.8.0
bin/zookeeper-server-start.sh config/zookeeper.properties
```
ZooKeeper, as of this version, is required for Kafka to work. ZooKeeper is responsibile for the overall management of Kafka cluster. It monitors the Kafka brokers and notifies Kafka if any broker or partition goes down, or if a new broker or partition goes up.


# 3 - Start the Kafka broker service
Start a new terminal.
Run the commands below. This will start the Kafka message broker service.
```
cd kafka_2.12-2.8.0
bin/kafka-server-start.sh config/server.properties
```

# 4 - Create a topic
You need to create a topic before you can start to post messages.

To create a topic named `news`, start a new terminal and run the command below.
```
cd kafka_2.12-2.8.0
bin/kafka-topics.sh --create --topic news --bootstrap-server localhost:9092
```
You will see the message: 'Created topic news.'

# 5 - Start Producer
You need a producer to send messages to Kafka. Run the command below to start a producer.
```
bin/kafka-console-producer.sh --topic news --bootstrap-server localhost:9092
```
Once the producer starts, and you get the '>' prompt, type any text message and press enter. Or you can copy the text below and paste. The below text sends three messages to kafka.
> Good morning

> Good day

> Enjoy the Kafka lab

# 6 - Start Consumer
You need a consumer to read messages from kafka.

Open a new terminal.

Run the command below to listen to the messages in the topic `news`.

```
cd kafka_2.12-2.8.0
bin/kafka-console-consumer.sh --topic news --from-beginning --bootstrap-server localhost:9092
```
You should see all the messages you sent from the producer appear here.

You can go back to the producer terminal and type some more messages, one message per line, and you will see them appear here.

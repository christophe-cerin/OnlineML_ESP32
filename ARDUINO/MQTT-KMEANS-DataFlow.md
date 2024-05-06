## Use of the MQTT protocol combined with the algorithm Kmeans with a Data Flow

### Functional architecture of the system
Our design is as follows, firstly we will inject data from the
.csv format in the same MQTT Broker which will send and receive the messages, i.e.
subscribe/publish.
**Connection and Disconnection**
MQTT uses persistent connections between clients and the broker, and for this exploits the
network protocols guaranteeing a good level of reliability such as TCP.
Before being able to send orders, the client must first register with the broker,
what is done with the CONNECT command
Various connection parameters can be exchanged such as client identifiers or the
desired persistence mode. The broker must confirm to the client that their REGISTRATION has been successful.
taken into account, or indicate that an error has occurred by returning a CONNACK accompanied
a return code.
When the client wants to disconnect, it first sends a DISCONNECT command to the
broker to inform him of his intention. Otherwise, the broker will consider the
disconnection as abnormal.

### Subscriptions and Publications
Each published message is necessarily associated with a subject, which allows its distribution to
subscribers. Topics can be organized in a tree hierarchy, thus subscriptions
may relate to filtering reasons. Subscription management is very simple and consists of
three essential commands:
1. **SUBSCRIBE** allows a customer to subscribe to a topic, once subscribed he will receive via
following all publications concerning this subject.
2. **UNSUBSCRIBE** gives the possibility of canceling a subscription, and thus no longer receiving the
subsequent publications.
3. **PUBLISH** initiated by a client, allows you to publish a message which will be transmitted by the broker
to potential subscribers. The same command will be sent by the broker to subscribers for
deliver the message.

**Topic and filtering reasons**
A subject is a UTF-8 string, which is used by the broker to filter messages to filter
messages for each connected client. A topic is made up of one or more topic levels.
Each topic level is separated by a slash. Here are some example topics:
concession / house / living room / sofa
Each subject must have 1 character to be valid and it can also contain spaces. There
slash alone is a valid topic. A customer can subscribe to several topics at once.
In order to offer an effective filtering system on subjects, it is possible to define a
tree structure using the / separator.
Two wildcards are reserved to represent one and more tree levels:
- \+ represents a tree level, so T1/T2/T3 can be mapped to
various filters such as T1 / T2 / +, T1 / + / T3 or T1 / + / +.
- \# represents as many levels as possible, and can only be used at the end of a filter pattern;
thus T1 / # will filter all topics published by the broker with the exception of special topics
starting with $.
Security
Three concepts are fundamental to MQTT security:
- Identification consists of naming the broker and the client to whom access rights are given.
The MQTT broker identifies itself to the client with its IP address and digital certificate.
The MQTT client uses the SSL protocol to authenticate the certificate sent by the broker
- Authentication seeks to mutually prove the identity of the client and the broker. A
client authenticates a broker using the SSL protocol. An MQTT broker authenticates a
client using SSL, password, or both.
- Authorization consists of managing the client's rights. Authorization is not part of the
MQTT protocol. It is provided by MQTT brokers. What is allowed or not depends
what the broker does.

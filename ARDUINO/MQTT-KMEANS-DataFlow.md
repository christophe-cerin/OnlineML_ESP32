## Use of the MQTT protocol combined with the algorithm Kmeans and a Data Flow

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

**Security**

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

**The MQTT client ID**

The MQTT protocol defines a "client identifier" - client ID that uniquely identifies a
client in a network. Simply put, when connecting to a broker, a client must specify
a unique string that is not currently in use and will not be used by another client that is
will connect to the MQTT broker.
Now let's try to understand the implications of two clients getting the same username.
customer. the MQTT broker monitors messages waiting to be sent to a running client
customer ID. Thus, if a client uses the quality of service QoS1, that is to say the message is
sure to arrive but it can do it several times or QoS2 so that the messages are not
sent in duplicate. MQTT provides the possibility of having at most 65535 messages pending with a
16-bit message identifier.

### Data Flow

The MQTT standard and binary packet format. The data is represented in 3 forms which
follow

1. Bits: they are labeled 7 to 0 without a byte
2. Integer data values that are 16 bits
3. character strings, they must be encoded in utf8 and prefixed by their length
on two bytes, these strings are limited to a length of 65,535 bytes (216 - 1).

In our system there will be the user who will be the external actor via an xtem terminal or console and
the MQTT broker.
The data flow will be as follows:

1. The user sends an authentication request
2. If the information is correct, the topics in which it will be loaded
3. Topics will be displayed in the user interface
4. Messages will be sent and received between the broker and the user according to the topics

## Design

Details of some essential algorithms
__________________________________________________________________________________
**Algorithm 1**: Authentication method
__________________________________________________________________________________
**Result**: Status

Read Username and Password settings;
Connection to the database;

**if** the line exists **then**

Status = connection;

**else**

Status = error;

**end**
________________________________________________________________________________
This previous pseudo algorithm represents the following authentication method:
1. The method takes two string type parameters which will be the name
username and password
2. We initialize the parameters received.
3. We connect to the database on the server.
4. If the line exists, we connect.
5. Otherwise an error message will be displayed
The return message sending pseudocode will be as follows:
__________________________________________________________________________________
**Algorithm 2**: Return Message Sending Method
__________________________________________________________________________________
**Result**: Message

Connect to the MQTT Broker;

**if** Connection = success **then**

x = true;

**else**

x = false;

**while** x == true **do**

wait for message;

**if** request received **then**
sub;
publish;
________________________________________________________________________________
The return message sending method will be as follows:
1. Connect to the MQTT Broker which will be on the server
2. Wait for the request message
3. Read predictions
4. read the learnings

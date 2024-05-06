## Use of the MQTT protocol combined with the algorithm
Kmeans with a Data Flow

### Functional architecture of the system
Our design is as follows, firstly we will inject data from the
.csv format in the same MQTT Broker which will send and receive the messages, i.e.
subscribe/publish.
Connection and Disconnection
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

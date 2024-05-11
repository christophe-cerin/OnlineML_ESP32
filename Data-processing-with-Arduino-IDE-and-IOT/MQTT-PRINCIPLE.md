## Tutorial on the MQTT protocol : description of the
technique with practical example of Mosquitto

MQTT protocol is a Machine to Machine (M2M) protocol widely used in IoT
(Internet of Things).
The MQTT protocol is a message-based protocol, extremely lightweight and for this reason,
it is adopted in IoT. Almost all IoT platforms support MQTT for
send and receive data from smart objects.
This tutorial provides an in-depth MQTT tutorial covering:
- how MQTT works
- MQTT messages
- how to use it in IoT projects
- how to configure Mosquitto MQTT to handle the MQTT protocol
There are several implementations for different IoT boards like Arduino, Raspberry, ESP32,
ESP8266, etc.
There are other IoT protocols used to implement IoT projects, but MQTT is one
most effective.

### Technical description of the MQTT protocol
The MQTT IoT protocol was developed around 1999. The main objective of this protocol was to
create a very bandwidth-efficient protocol. In addition, it is a very
energy efficient. For all these reasons, it is suitable for IoT.
This uses the publish-subscribe paradigm unlike HTTP based paradigm
request/response. It uses binary messages to exchange information with low
overload. It's very simple to implement and it's open. All these aspects contribute to its broad
adoption in IoT. Another interesting aspect is the fact that the MQTT protocol uses the TCP stack
as a transmission substrate.

### The role of the MQTT Broker and the MQTT client
As noted previously, the MQTT protocol implements a publish-
subscription. This paradigm dissociates a client which publishes a message (“publisher”) from other clients
who receive the message (“subscribers”). Additionally, MQTT is an asynchronous protocol, which means
that it does not block the client while it waits for the message.
Unlike HTTP, it is a primarily asynchronous protocol. Another one
interesting property of the MQTT protocol is that it does not require that the client ("subscriber") and
the editor are connected at the same time.

The key component of MQTT is the MQTT broker. The main task of the MQTT broker is to
send messages to MQTT clients (“subscribers”). In other words, the MQTT broker receives
messages from the publisher and distributes these messages to subscribers.

### The MQTT topic
While delivering messages, the MQTT broker uses the topic to filter clients
MQTT which will receive the message. The topic is a string and it is possible to combine topics into
creating subject levels.
A topic is a virtual channel that connects a publisher to its subscribers. The MQTT broker handles this topic. HAS
through this virtual channel, the publisher is decoupled from subscribers and MQTT clients (publishers or
subscribers) do not need to know each other to exchange data. This makes this protocol
highly scalable without direct dependence on the message producer ("publisher") and the
message consumer (“subscriber”).
### How to use MQTT protocol with Mosquitto?
Now we have an overview of MQTT and it's time to know how to use it using a
real example. There are several implementations of MQTT, in this example we will use
Mosquitto, an implementation developed by Eclipse. The first step is to install the
MQTT broker. We will install it on our computer. To install it we need to add the
repository that contains the application, so we can download it. Before adding the
repository, it is necessary to add the key to verify that the download package is
valid.
```
apt-get install mosquitto
```
The MQTT server (aka MQTT broker) is installed on our machine. This server is our broker
MQTT as specified above. Now we need to install the client, or in other words,
the publisher and the subscriber. In this example, we will install the client and the server on the same
Raspberry but you can install it on another pc/server or IoT card.
```
apt-get install mosquitto-clients
```
You can experiment with how to use MQTT by reading how to create an Arduino client
MQTT.

### How to send and receive an MQTT message?
We have installed and configured the client and server, now we can register a
subscribed to a specific topic and wait for an incoming message from an editor. To register a subscriber,
we will use this command: **Terminal 1**

```
mamadou@dugny:~/Arduino/mqtt/sketch_wifia212$ mosquitto_sub -d -t swa_news
Client (null) sending CONNECT
Client (null) received CONNACK (0)
Client (null) sending SUBSCRIBE (Mid: 1, Topic: swa_news, QoS: 0, Options:
0x00)
Client (null) received SUBACK
Subscribed (mid: 1): 0
```
As you can see, our subscriber is waiting for a message. In this example, we have
used a topic called swa_news. We will now send a message using a
MQTT editor that uses the same topic swa_news.
In the example, the MQTT publisher sends the "Hello Protocol" message. On the subscriber side, we
receive the message: **Terminal 2**


```
mamadou@dugny:~$ mosquitto_pub -d -t swa_news -m "Hello Protocol"
Client (null) sending CONNECT
Client (null) received CONNACK (0)
Client (null) sending PUBLISH (d0, q0, r0, m1, 'swa_news', ... (14 bytes))
Client (null) sending DISCONNECT
```
An important aspect to note is that MQTT is a simple protocol, so the message is clear and
everyone can read it. **Terminal 1**

```
mamadou@dugny:~/Arduino/mqtt/sketch_wifia212$ mosquitto_sub -d -t swa_news
Client (null) sending CONNECT
Client (null) received CONNACK (0)
Client (null) sending SUBSCRIBE (Mid: 1, Topic: swa_news, QoS: 0, Options:
0x00)
Client (null) received SUBACK
Subscribed (mid: 1): 0
Client (null) received PUBLISH (d0, q0, r0, m0, 'swa_news', ... (14 bytes))
Hello Protocol
Client (null) sending PINGREQ
Client (null) received PINGRESP
Client (null) sending DISCONNECT
```


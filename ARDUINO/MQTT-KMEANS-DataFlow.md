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
__________________________________________________________________________________
**Algorithm 3**: K-means algorithm
__________________________________________________________________________________
**Input**:

D ={x<sub>1</sub>, x<sub>2</sub>,x<sub>3</sub>, ... x<sub>n</sub>} // Data Entry

K // Desired number of clusters

**Output**:

K // Entering Clusters

**K-Means**:

Assignment of initial values for m<sub>1</sub>,m<sub>2</sub>, m<sub>3</sub>, ..... m<sub>k</sub>

**repeat**

each element x<sub>n</sub> to the clusters whose mean is closest; calculate a
new average for each cluster

**until**

the convergence criteria are met
______________________________________________________________________________


### MQTT Broker

Eclipse Mosquitto is an Open Source Message Broker (under EPL/EDL license) which
implements several versions of the MQTT protocol. Mosquitto is lightweight and suitable for all
devices, from low-power single-board computers to complete servers.
The MQTT protocol provides a lightweight method of running messaging using a
publication/subscription. this makes it suitable for Internet of Things messaging, for example with
low power sensors or mobile devices such as phones, computers
integrated or microcontrollers.
The Mosquitto project also provides a C library for implementing MQTT clients,
and the very popular online MQTT clients mosquitto_pub and mosquitto_sub.
Mosquitto is part of the Eclipse Foundation and is an iot project.eclipse.org is sponsored by
CEDALO.COM.

#### Subscribe to a topic / Suscrib to topic
To subscribe, call the client.Subscribe method with three parameters:

- topic: string with the subject of the subscription
- qos: 0 (fire-and-forget), 1 (resend if missed) or 2 (make sure it is not received
only once)
- callback: a function to call when a message from this subject is received. It can be nil
so only the default handler will be called

#### Publish to a Topic
To publish a message, call the client.Publish method. It receives four parameters:

- topic: same topic as before, send a timestamp before disconnecting
- qos: 0 (fire-and-forget), 1 (resend if missed) or 2 (make sure it is only received once)
- retained: boolean indicating whether the message must be retained by the server
- payload: message to publish under the subject

## Intervention and Adaptation of the K-Means algorithm
### What is Kmeans?
Kmeans is one of the easiest access algorithms among clustering algorithms. Kmeans
is one of the machine learning algorithms without filling and separating the data into
specific cluster. The remarkable feature is to be able to choose the number of clusters in
which data is to be separated.

### Kmeans algorithm
The Kmeans algorithm defines k representative points and ensures that the data belongs to the
nearest representative point. About each representative point, it calculates the average of the
data that belongs to this point. The calculated point becomes a new representative point and
again it makes the data belong to the closest ones. Until you can't find the
change of state, simply repeat the cycle.
In a word, it can be summarized as follows after defining the first representative points,
- make the data belong to the nearest representative point
- make the center of gravity of the assigned data a new representative point
- repeat these 2 steps above until the state does not change

### Characteristics of the implemented Kmeans algorithm
- Initialize the representative points with a fixed random number
- Use Euclidean distance as a distance function
- Truncate it when it cannot observe any label changes by upgrade
- Most of the code is the fit() method. The fit() method takes the target data from
clustering and the number of clusters as arguments. At first it stores the target data
in the structure.
- Kmeans first prepares the k representative points. k is the number that we can choose
as number of clusters. And these are updated by the data. In practice, the
Initial values of these representative points are very important. This time we have to give
the initial values by a random number whose range is from the minimum of variables
explanatory information as relevant as possible.
- Part of the code calculates the distance between the data and the representative points and
the label of the nearest representative point is given to the data. This time the clue
representative points is used as a label.
- The Kmeans algorithm: On the first half, it updates the representative points and on the
second half it updates the data label. Concretely, updating the points
representative defines the centroid of the data that belongs to the representative point
as a new updated point. Updating labels sets a new label
by calculating the distance between the data and the representative points. Usually, the
truncation is performed when the change state becomes stable, meaning that
the label changes according to the change of the representative points, the change of
distance, etc. This time the concise method chooses truncates when updating the point
representative does not change any labels, to reduce code size. In fact, of this
way, if the initialized representative points are so biased, some labels contain
too much data and a single update is not enough to change the label and
learning can be truncated. So, to use this algorithm, it is better to think about
how to give initial values and how to truncate.

## Send and receive data over the serial channel of the Arduino IDE
We will learn how to use the serial route with Arduino. We will see how to send then
receive information with the computer, finally we will do some exercises to check that
you understood everything.

### Preparing the serial channel
How to communicate information from a microcontroller or Arduino board, Seeed WIOT,
ESP-WROOM-32D ... to the computer and vice versa.

**On the computer side**
To be able to use computer communication, Arduino development environment
offers a basic tool for communicating, just click on the Tools/Monitor bar
Series or on the Series Monitor icon in the Horizontal Icon Bar a window opens: this is
the Serial Terminal. In this window, you can send messages on the serial channel of
the computer which is emulated by the Arduino IDE; receive messages that the Arduino IDE tells you
send ; and adjust two or three parameters such as the communication speed with the Arduino and
AutoScroll which creates the text automatically.

**On the program side**

##### The Serial object
To use the serial channel and communicate with our computer, we use an object (an output of
variable but more advanced) which is natively integrated into the Arduino assembly: the Serial object.
This object gathers information (speed, data bits, etc.) on what a serial channel is.
for Arduino. Thus, there is no need for the programmer to recreate all the protocol otherwise we would have had to
write the WHOLE protocol, such as “Write a high bit for 1 ms, then 1 low bit for 1 ms,
then the character “a” in 8 ms...

##### The Setup
To begin, we initialize the Serial object each time. With the aim of creating a
communication between the computer and the card with the microcontroller, it is necessary to declare a
communication and define the speed at which these two devices will communicate. If this
speed is different, the Arduino IDE will not understand what the computer and vice versa. This
adjustment is made in the **setup** function using the **begin()** function of the **Serial** object

```
void setup() {
// we start the connection
// by setting it to a speed of 9600 bits per second.
169Serial.begin(115200);
while (!Serial) {
}
}
```

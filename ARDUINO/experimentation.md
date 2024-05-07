## The ESP-WROOM-32D Microcontroller, MQTT
Broker 10.10.6.228 and the Arduino-IDE on the computer
portable

### Configuring the MQTT Broker server

```
mamadou@port-lipn12:~/Arduino/libraries/ESP32WROOMDAMODULE_Arduino_KMEANS_MQTT/src/
data$ cat /etc/mosquitto/mosquitto.conf
# Place your local configuration in /etc/mosquitto/conf.d/
#
# A full description of the configuration file is at
# /usr/share/doc/mosquitto/examples/mosquitto.conf.example
#
listener 1883 10.10.6.228
#listener 8883 10.10.6.228
#allow_anonymous false
allow_anonymous true
#password_file /etc/mosquitto/passwd
pid_file /run/mosquitto/mosquitto.pid
persistence true
persistence_location /var/lib/mosquitto/
log_dest file /var/log/mosquitto/mosquitto.log
include_dir /etc/mosquitto/conf.d
```

### Data management, data/celsius in csv format

```
mamadou@port-lipn12:~/Arduino/libraries/ESP32WROOMDAMODULE_Arduino_KMEANS_MQTT/src/
data$ cat test-29062021-broker-10.10.6.228-500points-Portable-Dell-pub.sh
name=`date +%d%m%Y%H%M%N.csv`
#for i in {1..12256}; do
## Avec l'Ordinateur maintenir que 500 points
for i in {1..500}; do
#for i in {1..286}; do
sleep 0.05s
#foo=`gdate +%H%M%S%N`;
#foo=`date +%H%M%S%N`;
#i=`expr $foo / 1000`;
#j=`gshuf -i 1-10000000 -n 1`;
i=`shuf -i 1-10000000 -n 1`;
j=`shuf -i 1-10000000 -n 1`;
#echo "$i,$j"
/usr/bin/mosquitto_pub -h 10.10.6.228 -t date/celsius -m "$i,$j"
#/usr/bin/mosquitto_pub -h 10.10.6.228 -t date/celsius -m "$i,$j"
echo "$i,$j" >> ${name}
done
# publish the signal to terminate
/usr/bin/mosquitto_pub -h 10.10.6.228 -t date/celsius -m "0.0,0.0"
#publish the file name of generate data
/usr/bin/mosquitto_pub -h 10.10.6.228 -t final/final -m "${name},${name}"
```
## Arduino program for receiving messages and printing the message on the serial port

```
/* ESP32_MQTT_WFI_RECEIVING_MESSAGE
ArduinoMqttClient - WiFi Simple Receive
This example connects to a MQTT broker and subscribes to a single topic.
When a message is received it prints the message to the Serial Monitor.
The circuit:
- esp32, Arduino MKR 1000, MKR 1010 or Uno WiFi Rev2 board
This example code is in the public domain.
*/
#include <ArduinoMqttClient.h>
#if defined(ARDUINO_SAMD_MKRWIFI1010) || defined(ARDUINO_SAMD_NANO_33_IOT) ||
defined(ARDUINO_AVR_UNO_WIFI_REV2)
#include <WiFiNINA.h>
#elif defined(ARDUINO_SAMD_MKR1000)
#include <WiFi101.h>
#elif defined(ARDUINO_ARCH_ESP8266)
#include <ESP8266WiFi.h>
#elif defined(ARDUINO_PORTENTA_H7_M7) || defined(ARDUINO_NICLA_VISION) ||
defined(ARDUINO_ARCH_ESP32) || defined(ARDUINO_GIGA)
#include <WiFi.h>
#endif
#include <PubSubClient.h> //Librairie pour la gestion Mqtt
///////please enter your sensitive data in the Secret tab/arduino_secrets.h
char ssid[] = SECRET_SSID; // your network SSID (name)
char pass[] = SECRET_PASS; // your network password (use for WPA, or use as key for WEP)

// To connect with SSL/TLS:
// 1) Change WiFiClient to WiFiSSLClient.
// 2) Change port value from 1883 to 8883.
// 3) Change broker value to a server with a known SSL/TLS root certificate
// flashed in the WiFi module.
WiFiClient wifiClient;
MqttClient mqttClient(wifiClient);
//const char broker[] = "test.mosquitto.org";
const char broker[] = "10.10.6.228";
int port = 1883;
const char topic[] = "date/celsius";
void setup() {
//Initialize serial and wait for port to open:
Serial.begin(115200);
while (!Serial) {
; // wait for serial port to connect. Needed for native USB port only
}
// attempt to connect to Wifi network:
WiFi.begin(ssid, pass);
//while ( WiFiMulti.run() != WL_CONNECTED ) {
while (WiFi.status() != WL_CONNECTED) {
delay (500 );
Serial.print ("Tentative de connection du Micro-Contrôleur ESP32 au Réseau Wifi : ");
Serial.println( ssid);
}
Serial.println("You're connected to the network");
Serial.println();
Serial.println(WiFi.macAddress());
Serial.print("Adresse IP du Micro-Contrôleur ESP32 : ");
Serial.println(WiFi.localIP());
// You can provide a unique client ID, if not set the library uses Arduino-millis()
// Each client must have a unique client ID
// mqttClient.setId("clientId");
// You can provide a username and password for authentication
// mqttClient.setUsernamePassword("username", "password");
Serial.print("Attempting to connect to the MQTT broker: ");
Serial.println(broker);
if (!mqttClient.connect(broker, port)) {
Serial.print("MQTT connection failed! Error code = ");
Serial.println(mqttClient.connectError());
while (1);
}
Serial.println("You're connected to the MQTT broker!");
Serial.println();
Serial.print("Subscribing to topic: ");
Serial.println(topic);
Serial.println();
// subscribe to a topic
mqttClient.subscribe(topic);
// topics can be unsubscribed using:
// mqttClient.unsubscribe(topic);
Serial.print("Waiting for messages on topic: ");
Serial.println(topic);
Serial.println();
}
void loop() {
int messageSize = mqttClient.parseMessage();
if (messageSize) {
// we received a message, print out the topic and contents
Serial.print("Received a message with topic '");
Serial.print(mqttClient.messageTopic());
Serial.print("', length ");
Serial.print(messageSize);
Serial.println(" bytes:");
// use the Stream interface to print the contents
while (mqttClient.available()) {
Serial.print((char)mqttClient.read());
}
Serial.println();
Serial.println();
}
}
```


<img alt="the GPIO pins of the ESP32-WROOM-32D" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/images/generdata.png" width=50% height=50%  title="the GPIO pins of the ESP32-WROOM-32D"/>

###### **The GPIO pins of the ESP32-WROOM-32D**

<img alt="the GPIO pins of the ESP32-WROOM-32D" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/images/arduinoreceivmessag.png" width=30% height=30%  title="the GPIO pins of the ESP32-WROOM-32D"/>

###### **The GPIO pins of the ESP32-WROOM-32D**

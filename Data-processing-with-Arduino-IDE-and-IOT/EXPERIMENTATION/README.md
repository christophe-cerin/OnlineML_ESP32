## The ESP-WROOM-32D Microcontroller, MQTT
Broker IP Address : 10.10.6.228 and the Arduino-IDE on the computer
portable

### Configuring the MQTT Broker server

```
$ cat /etc/mosquitto/mosquitto.conf
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
$ cat test-29062021-broker-10.10.6.228-500points-Portable-Dell-pub.sh
name=`date +%d%m%Y%H%M%N.csv`
#for i in {1..12256}; do
## Avec l'Ordinateur maintenir que 500 points
for i in {1..500}; do
sleep 0.05s
    i=`shuf -i 1-10000000 -n 1`;
    j=`shuf -i 1-10000000 -n 1`;
    /usr/bin/mosquitto_pub -h 10.10.6.228 -t date/celsius -m "$i,$j"
echo "$i,$j" >> ${name}
done
# publish the signal to terminate
/usr/bin/mosquitto_pub -h 10.10.6.228 -t date/celsius -m "0.0,0.0"
/usr/bin/mosquitto_pub -h 10.10.6.228 -t final/final -m "${name},${name}"
```
## Arduino program for receiving messages and printing the message on the serial port

```
/* 
M. SOW
12/09/2023
ArduinoMqttClient - WiFi Simple Receive
This example connects to a MQTT broker and subscribes to a two topics.
When the messages are received it prints the messages to the Serial Monitor.
The circuit:
- ESP32, Arduino MKR 1000, MKR 1010 or Uno WiFi Rev2 board
This example code is in the public domain.
- Open a Terminal and execute the program "test-29062021-broker-10.10.6.228-500points-Portable-Dell-pub.sh"
- We can have a response on the Serial Monitor of Arduino IDE
*/

#include <ArduinoMqttClient.h>
#if defined(ARDUINO_SAMD_MKRWIFI1010) || defined(ARDUINO_SAMD_NANO_33_IOT) || defined(ARDUINO_AVR_UNO_WIFI_REV2)
#include <WiFiNINA.h>
#elif defined(ARDUINO_SAMD_MKR1000)
#include <WiFi101.h>
#elif defined(ARDUINO_ARCH_ESP8266)
#include <ESP8266WiFi.h>
#elif defined(ARDUINO_PORTENTA_H7_M7) || defined(ARDUINO_NICLA_VISION) || defined(ARDUINO_ARCH_ESP32) || defined(ARDUINO_GIGA)
#include <WiFi.h>
#endif

#include <PubSubClient.h> //Librairie pour la gestion Mqtt
///////please enter your sensitive data in the Secret tab/arduino_secrets.h
char ssid[] = SECRET_SSID; // your network SSID (name)
char pass[] = SECRET_PASS; // your network password (use for WPA, or use as key for WEP)

WiFiClient wifiClient;
MqttClient mqttClient(wifiClient);

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
## Results

<img alt="Console bash" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/Data-processing-with-Arduino-IDE-and-IOT/images/generdata.png" width=70% height=70%  title="Console bash"/>

###### **Figure 1 : In a terminal you have to run the program to generate**

<img alt="Arduino IDE serial monitor" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/Data-processing-with-Arduino-IDE-and-IOT/images/arduinoreceivmessag.png" width=60% height=60%  title="Arduino IDE serial monitor"/>

###### **Figure 2 : Result obtained in the arduino ide serial monitor**

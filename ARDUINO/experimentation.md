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

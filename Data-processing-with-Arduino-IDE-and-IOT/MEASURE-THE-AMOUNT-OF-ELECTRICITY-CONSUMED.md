## Let's measure the amount of electricity consumed by the ESP32-VROOM-32D during data processing with Arduino

### Necessary material :
 
- the current source to power the ESP32 card will be PPK2
- the multimeter will be the PPK2 using the nrfconnect software
- the Arduino IDE on the Laptop PC needs to detect the ESP32-VROOM-32D card through the serial port of the console, the CP2102N module or USB-TTL converter will be the suitable hardware.

### Handling: follow the steps as indicated

We need 2 USB ports on our laptop
USB port n°1 of the laptop

- connect the Laptop PC and the ESP32-VROOM-32D using the microUSB/USB cable
- load your sketch or program from the Arduino IDE then disconnect the ESP32 card and replace it with the PPK2
- the PPK2, connected to the Laptop via the miniUSB/USB cable, will be mounted on the device /dev/ttyACM0
- now connect PPK2 and ESP32, two female grove cables will be useful for this. The red cable will connect the positive terminal of the 3.8 V voltage difference on ESP32 to the VOUT (+) terminal on PPK2. The black cable will connect the negative terminal of the 3.8 V voltage difference on ESP32 to the GND(-) terminal on PPK2
USB port no. 2 of the laptop
- connect the CP2102N module to the Laptop, it will be mounted on the device /dev/ttyUSB0
- connect the RX, TX, GND GPIO of the ESP32 card to the RX, TX, GND pins of the CP2102N module with 3 female grove cables.



<img alt="Schematic representation of all hardware and connection principle" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/Data-processing-with-Arduino-IDE-and-IOT/images/connect-PC-PPK2-CP2102-ESP32.png" width=60% height=60%  title="Schematic representation of all hardware and connection principle"/>

###### **Schematic representation of all hardware and connection principle**

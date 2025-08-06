
# Storing Data from Serial Port to Disk

**Abstract**

This project aims to display real-time temperature and humidity on an LCD screen and then save this data to a local or remote computer.
To achieve this, we're using an  Arduino Uno R4 Wifi board and the  Arduino IDE to program the system. Additional hardware includes a DHT22 sensor , a potentiometer, a 16x2 LCD screen, and a breadboard.
Traditionally, data is stored on an SD card, which is why the `SD.h` library is commonly included. This would allow for creating a read/write file to record temperature and humidity in two columns, provided the SD card's CS pin is correctly configured in the program.
However, using flash memory as a storage solution isn't considered due to its very limited capacity.
For a data logging system with minimal constraints, we are prioritizing direct storage on the laptop's hard drive or on a remote computer.

## 1. Hardware Setup: Connecting the Screen, Potentiometer, and DHT22 Sensor

Here's a breakdown of the components you'll need and how to connect them for your project :

Components
 - Arduino Uno R4 Wifi (Your main microcontroller unit)
 - 16x2 LCD Screen (16 pins)
 - DHT22 Sensor (for temperature and humidity)
 - Potentiometer
 - 1 k Ohm Resistor
 - Breadboard (Planche de Câble)
 - Micro USB-C to USB2 Cable

Wiring Instructions
 1 k Ohm Resistor
  - Connect this resistor for overvoltage protection of the LCD screen.
 Potentiometer
  - The third pin of the potentiometer connects to the V0 pin of the LCD screen.
 DHT22 Sensor
  - Green cable (for data/display) connects to digital pin 13 on the Arduino Uno.
  - Red cable (for power) connects to the positive rail of the breadboard.
  - Black cable (for ground) connects to the negative rail of the breadboard.
 16x2 LCD Screen
   - LCD Pin - (Negative) connects to Arduino Uno R4 Wifi GND digital pin.
   - LCD Pin + (Positive) connects to Arduino Uno R4 Wifi 5V digital pin.
   - LCD D7 pin connects to Arduino Uno R4 Wifi digital pin 2.
   - LCD D6 pin connects to Arduino Uno R4 Wifi digital pin 3.
   - LCD D5 pin connects to Arduino Uno R4 Wifi digital pin 4.
   - LCD D4 pin connects to Arduino Uno R4 Wifi digital pin 5.
   - LCD Enable pin connects to Arduino Uno R4 Wifi digital pin 11.
   - LCD R/W (Read/Write) pin connects to Arduino Uno R4 Wifi GND digital pin.
   - LCD RS (Register Select) pin connects to Arduino Uno R4 Wifi digital pin 12.
   - LCD VCC/VDD pin connects to Arduino Uno R4 Wifi 5V digital pin.
   - LCD VSS pin connects to Arduino Uno R4 Wifi GND digital pin.

(Note: LCD pins D0, D1, D2, D3, and V0 are mentioned but their connections are implied or handled by the potentiometer connection for V0.)


<picture>
<center>
<img alt="Matrice X" align="center" src="https://github.com/christophe-cerin/OnlineML_ESP32/blob/main/ARDUINO/Data-processing-with-Arduino-IDE-and-IOT/ARDUINO-UNO-R4-WIFI/images/schemaUnoDht22LcdPot.png" width=40% height=40%  title="unoDht22Lcd"/>
</center>
</picture>

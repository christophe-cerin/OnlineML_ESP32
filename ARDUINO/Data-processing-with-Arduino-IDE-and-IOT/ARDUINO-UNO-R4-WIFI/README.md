
# Storing Data from Serial Port to Disk

*Abstract*
This project aims to display real-time temperature and humidity on an LCD screen and then save this data to a local or remote computer.
To achieve this, we're using an  Arduino Uno R4 Wifi board and the  Arduino IDE to program the system. Additional hardware includes a DHT22 sensor , a potentiometer, a 16x2 LCD screen, and a breadboard.
Traditionally, data is stored on an SD card, which is why the `SD.h` library is commonly included. This would allow for creating a read/write file to record temperature and humidity in two columns, provided the SD card's CS pin is correctly configured in the program.
However, using flash memory as a storage solution isn't considered due to its very limited capacity.
For a data logging system with minimal constraints, we are prioritizing direct storage on the laptop's hard drive or on a remote computer.

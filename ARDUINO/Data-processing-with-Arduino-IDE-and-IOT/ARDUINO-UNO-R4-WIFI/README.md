
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

<figure>
   <img alt="Matrice X" align="center" src="https://github.com/christophe-cerin/OnlineML_ESP32/blob/main/ARDUINO/Data-processing-with-Arduino-IDE-and-IOT/ARDUINO-UNO-R4-WIFI/images/schemaUnoDht22LcdPot.png" width=40% height=40%  title="unoDht22Lcd"/>
    <figcaption><b>Figure 1:</b> Wiring diagrams</figcaption>
</figure>

## 2. Libraries Installed in Arduino IDE

Below is a list of the libraries used in this project, along with their primary functions:

|Library | Function |
| --- | --- |
| DHT22 (by dvarrel), SimpleDHT (by winlin) | Temperature & Humidity Sensor |
| LiquidCrystal (by Adafruit and Arduino) | LCD Screen Display |
| DHT sensor library (by Adafruit) | Temperature & Humidity Sensor |
| Adafruit AM2320, sensor library (by Adafruit) | Temperature & Humidity Sensor |

## 3. Two Essential Programs for Our Method

Our approach relies on the use of two distinct programs. The first program is an Arduino program (file: tempHumLcdDHThdstatus.ino), whose data is read via the serial port of the Arduino IDE.

Description of the tempHumLcdDHThdstatus.ino program:
 - Main Functionality: Real-time display of ambient temperature and humidity on an LCD screen, as well as saving this data to a computer via the serial port.
 - Objective: This program is designed to display data (temperature and humidity readings) measured by a DHT22 sensor on the Arduino IDE's serial monitor. The DHT22 sensor is connected to digital pin 13 of an Arduino Uno R4 Wifi board.
  - Author: M SOW
  - Creation Date: 03.07.2025
  - Libraries Used: LiquidCrystal.h (for the LCD screen) and SimpleDHT.h (for the DHT sensor).
  - Specific Arduino Script Enhancements:
   - Uses the read2() function for more precise floating-point values.
   - Improved error handling with direct display on the LCD screen.
   - Addition of Serial.flush() to ensure all data is sent.
   - More robust CSV output format, including an error status.
   - Display of values with one decimal place.
     
Recommended Serial Monitor Configuration :
 - Option 1: Open the Arduino IDE's serial monitor with a baud rate of 115200, then click "Save output" to save the data.
 - Option 2: Use an external program such as a console/terminal (on Linux/Mac) with a Python script to capture the data streamed over the serial port.

Description of the Arduino Program tempHumLcdDHthdstatus.ino

This Arduino program is designed to read temperature and humidity data from a DHT22 sensor, display it on a 16x2 LCD screen, and simultaneously send it to the Arduino IDE's serial monitor (or to a Python script) for PC logging.

Main Objective
 - Display real-time ambient temperature and humidity on an LCD screen.
 - Record this data on a computer via the serial port.
Author and Date
 - Author: Mamadou SOW
 - Created on: 03.07.2025
Libraries Used
The program includes two essential libraries:
 - <LiquidCrystal.h>: Used to control the 16x2 LCD screen.
 - <SimpleDHT.h>: Facilitates reading data from the DHT22 sensor.
Pin and Object Definitions
Constants define the Arduino pins connected to the LCD screen and the DHT22 sensor:
 - rs, en, d4, d5, d6, d7: Pins for the LCD screen interface.
 - POTENTIOMETRE (A0): Analog pin for the potentiometer (although not used in the data reading/sending logic, it is declared).
 - pinDHT22 (13): Digital pin to which the DHT22 sensor is connected.
Two objects are instantiated:
 - LiquidCrystal lcd(rs, en, d4, d5, d6, d7): The LCD object is created with the defined pins.
 - SimpleDHT22 dht22: The object for the DHT22 sensor is created.
setup() Function
This function executes once when the Arduino starts:
 - 1. Serial.begin(115200): Initializes serial communication at a baud rate of 115200.
 - 2. while(!Serial): Waits for the serial connection to be established (useful for boards that reset when the serial monitor is opened).
 - 3. pinMode(POTENTIOMETRE, INPUT): Configures the potentiometer pin as an input.
 - 4. lcd.begin(16,2): Initializes the LCD screen with 16 columns and 2 rows.
 - 5. Displays a startup message ("Demarrage...") on the LCD screen for 2 seconds, then clears it.
 - 6. Serial.println("timestamp_ms,temperature_C,humidity_pct,status"): Sends the column header to the serial monitor. This CSV format is crucial for data logging.
 - 7. Serial.flush(): Ensures the header is sent immediately.
envoyerVersPC() Function
This custom function is responsible for sending data to the PC via the serial port, in CSV format:
 - It takes temperature (temp), humidity (hum), and an optional status (defaulting to "OK") as parameters.
 - Serial.print(millis()): Sends the time elapsed since the Arduino started in milliseconds.
 - Serial.print(temp, 1) and Serial.print(hum, 1): Send temperature and humidity with one decimal place.
 - Serial.println(status): Sends the reading status (e.g., "OK" or "ERROR_XX").
 - Serial.flush(): Forces immediate data transmission to ensure it doesn't remain in the buffer.
loop() Function
This function runs continuously after setup():
- 1. DHT22 Sensor Reading:
  - dht22.read2(pinDHT22, &temperature, &humidity, NULL): Attempts to read temperature and humidity from the DHT22 sensor.
  - Error Handling: If the reading fails (err != SimpleDHTErrSuccess), an error message is displayed on the LCD screen with the error code, and an error line is sent to the PC via envoyerVersPC(0, 0, "ERROR_" + String(err)). The program waits 2 seconds before retrying.
- 2. LCD Display (if successful):
  - Clears the screen.
  - Displays the current temperature and humidity on both lines of the LCD screen.
- 3. Send to PC:
  - envoyerVersPC(temperature, humidity): Calls the function to send the read data (temperature, humidity, and "OK" status) to the PC.
- 4. delay(3000): The program pauses for 3 seconds before the next reading, meaning data is read and sent every 3 seconds.




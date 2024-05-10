
## IDE (Integrated Development Environment)

The integrated development environment (Integrated Development Environment abbreviated IDE) is
a software application that helps programmers develop software code efficiently. A
IDE typically includes a graphical interface to launch the different tools, an editor for
source code, a compiler, a debugger, and often a tool for building
software graphical interfaces

### Arduino IDE

Makers, students, and professionals use the classic Arduino IDE (environment
integrated development) since the birth of Arduino. The Arduino IDE 2.0 is an improvement
classic IDE, with increased performance, improved user interface and
many new features, such as auto-completion, a built-in debugger and
synchronization of sketches with Arduino Cloud.
The new major version of the Arduino IDE is faster and even more powerful! In addition to a
more modern editor and a more responsive interface, it offers auto-completion, navigation
in code and even a live debugger.

There are several methods for programming an ESP32 card, 
but for reasons of standardization and to limit the number of software, 
we will use the Arduino IDE tool which is quite easy to use. 
The Arduino card is very easily programmable with this software,
however, it will be necessary to make some preparations to be able to program the ESP32,
in particular, install the ESP32 card itself. Download and install the Arduino IDE from the official [Arduino.cc](https://www.arduino.cc/en/software) Website.

## Preparing the ESP-WROOM-32D microcontroller 

To develop applications for ESP-WROOM-32D, you needed :

- PC loaded with Ubuntu 22.04.1  LTS operating system
- Toolchain to create an application for ESP32
- ESP-IDF or Arduino which essentially contains the API for ESP32 and scripts to do
operate toolchain
- A text editor to write programs (Projects) in C for example Eclipse
- The ESP32 board itself and a USB cable to connect it to the PC
  
<!--
![Microcontrôleur ESP32-DevkitC équipé de WROOM-32](https://github.com/christophe-cerin/OnlineML_ESP32/blob/main/ARDUINO/images/carte-esp32-wroom-32d.png)
-->

<img alt="the GPIO pins of the ESP32-WROOM-32D" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/Data-processing-with-Arduino-IDE-and-IOT/images/carte-esp32-wroom-32d.png" width=30% height=30%  title="the GPIO pins of the ESP32-WROOM-32D"/>


###### **The GPIO pins of the ESP32-WROOM-32D**

## Presentation of the ESP-WROOM-32D microcontroller

ESP32-DevKitC on a turnkey basis is an entry-level ESP32 development board
range, it is also a small electronic card, called a microcontroller, which is easy to take
in hand. On the ESP32 board, the pins are pinned for connection and use
easily. The DevKit can be used to evaluate ESP-WROOM-32 modules and ESP32-D0WD chips.
- Flash / PSRAM: 4 MB flash,
- Interface: I/O,
- USB,
- User interface: button, LED
Related products: ESP-WROOM-32

There are a multitude of ESP32 boards with different pin placements. Standard serial ports on
gpio3 (RX) and gpio1 (TX) connections are used in series to communicate with Arduino IDE and to be connected to the
CP2102.

### Special operation of certain ESP32 pins

Development boards based on an ESP32 generally have 33 pins apart from those for the power supply.
Some GPIO pins have somewhat particular functions:
- If your ESP32 card has pins GPIO6, GPIO7, GPIO8, GPIO9, GPIO10, GPIO11, you should especially not
Do not use them because they are connected to the flash memory of the ESP32: if you use them the ESP32 will not work.
- The GPIO1 (TX0) and GPIO3 (RX0) pins are used to communicate with the computer in UART via USB.
If you use them, you will no longer be able to upload programs to the card or use the serial monitor via the
USB port. They can be useful for programming the board without using USB with an external programmer.
Fortunately, there are other UART interfaces available.
-  GPIO36 (VP), GPIO39 (VN), GPIO34, GPIO35 pins can be used as input only. They don't have
no integrated internal pullup and pulldown resistors either (We cannot use pinMode(36,
INPUT_PULLUP) or pinMode(36, INPUT_PULLDOWN) ).
- Some pins have a special role when starting the ESP32. These are the Strapping Pins. They are used for
put the ESP32 in BOOT mode (to execute the program written in flash memory) or FLASH mode
(to upload the program to flash memory). Depending on the tension present at the edge of these
pins, the ESP32 will boot either in BOOT mode or in FLASH mode. The strapping pins are the pins
GPIO0, GPIO2, GPIO12 (MTDI) and GPIO15 (MTDO). You can use them, but you just have to be careful
when a logic state (3.3V or 0V) is imposed with an external pullup or pulldown resistor.
- When booting the ESP32, for a short period, certain pins quickly change logical states (0V
→ 3.3V). You may have weird bugs with these pins: for example, a relay that activates
temporarily. The faulty pins are as follows:
- GPIO 1: Send the ESP32 boot logs via the UART
- GPIO 3: Voltage of 3.3V during boot
- GPIO 5: Sends a PWM signal during boot
- GPIO 14: Sends a PWM signal during boot
- GPIO 15: Sending the ESP32 boot logs via the UART
- The EN pin allows you to control the ignition status of the ESP32 via an external wire. It is connected to the EN button on the
map. When the ESP32 is turned on, it is at 3.3V. If we connect this pin to the ground, the ESP32 is turned off. We can use it
when the ESP32 is in a case and you want to be able to turn it on/off with a switch.
- The rest of the GPIO pins have no particular restrictions.
<!--
  ![Microcontrôleur ESP32-DevkitC équipé de WROOM-32](https://github.com/christophe-cerin/OnlineML_ESP32/blob/main/ARDUINO/images/carte-ESPWROOM32D.png)
  -->

  <img alt="ESP32-DevkitC microcontroller equipped with WROOM-32D" src="https://github.com/christophe-cerin/OnlineML_ESP32/blob/main/ARDUINO/images/carte-ESPWROOM32D.png" width=60% height=60%  title="ESP32-DevkitC microcontroller equipped with WROOM-32D "/>


###### **ESP32-DevkitC microcontroller equipped with WROOM-32D** 

  ### ESP32 peripherals
To interact with modules, sensors, or electronic circuits, the ESP32, like any microcontroller, has a
multitude of peripherals. They are also much more numerous than on a classic Arduino Uno board.
The ESP32 has the following peripherals:

- 3 UART interfaces
- 2 I2C interfaces
- 3 SPI interfaces
- 16 PWM outputs
- 10 capacitive sensors
- 18 analog inputs (ADC)
- 2 DAC outputs
  
Some peripherals are already used by the ESP32 during its basic operation. There is therefore in reality less
of possible interfaces for each device.

### The UART on the ESP32
UART is the serial protocol that allows data to be easily exchanged between 2 devices. On the ESP32 3 bus
UARTs are available: UART0, UART1 and UART2. They can be used to communicate with a sensor, a
Arduino, a Raspberry Pi, a computer...

- UART0 is by default on the GPIO1(TX0) and GPIO3(RX0) pins of the ESP32, it is used to communicate
with the computer via the serial monitor. It is also the one used to flash the ESP32 card. In general,
we use it to display messages in the console with Serial.println().
- To use UART2, simply add Serial2.begin() in the setup() function and use the function
Serial2.println() to send messages. By default, the UART2 bus is on pins GPIO16(RX2) and
GPIO17(TX2) but you can change them (useful with a Wrover module) during setup.
- UART1 is by default on the pins used by the ESP32 flash. However, it can be used thanks to the “GPIO
matrix” of the ESP32 by choosing the pins you want. So this code allows you to have a serial link on the pins
GPIO14 and GPIO12 using the UART1 bus.



## ESP32 peripherals
To interact with modules, sensors or electronic circuits, the ESP32, like any microcontroller, has a
multitude of peripherals. They are also much more numerous than on a classic Arduino Uno board.
The ESP32 has the following peripherals:

- 3 UART interfaces
- 2 I2C interfaces
- 3 SPI interfaces
- 16 PWM outputs
- 10 capacitive sensors
- 18 analog inputs (ADC)
- 2 DAC outputs
  
Some peripherals are already used by the ESP32 during its basic operation. There is therefore in reality less
of possible interfaces for each device.

## The UART on the ESP32
UART is the serial protocol that allows data to be easily exchanged between 2 devices. On the ESP32 3 bus
UARTs are available: UART0, UART1 and UART2. They can be used to communicate with a sensor, a
Arduino, a Raspberry Pi, a computer...

- UART0 is by default on the GPIO1(TX0) and GPIO3(RX0) pins of the ESP32, it is used to communicate
with the computer via the serial monitor. It is also the one used to flash the ESP32 card. In general,
we use it to display messages in the console with Serial.println() .
- To use UART2, simply add Serial2.begin() in the setup() function and use the function
Serial2.println() to send messages. By default, the UART2 bus is on pins GPIO16(RX2) and
GPIO17(TX2) but you can change them (useful with a Wrover module) during setup.
- UART1 is by default on the pins used by the ESP32 flash. However, it can be used thanks to the “GPIO
matrix” of the ESP32 by choosing the pins you want. So this code allows you to have a serial link on the pins
GPIO14 and GPIO12 using the UART1 bus.

## ARDUINO IDE and the first Hello World project
### Add the ESP32 card to the Arduino IDE card base

In the Arduino IDE:
- **Open File Menu > Preferences**
- Enter the following address in the field
  
Additional Map Manager URLs:
```
http://arduino.esp8266.com/stable/package_esp8266com_index.json,
https://dl.espressif.com/dl/package_esp32_index.json,
https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
```
### Install ESP32 board in Arduino IDE

Still in the Arduino IDE:
- **Open menu Tools > Map Type > Map Manager**
- Search for ESP32 package
- Install the package: select the latest version available and click on Install

### Select ESP32 card
All that remains is to define the ESP32 card so that the program is correctly sent to the
map. Still in the Arduino IDE:

- **Open menu Tools > Board Type > esp32 > ESP-WROOM DA Module**
- **Open menu Tools > Port > /dev/ttyUSB0**
- Then choose the ESP32 card that corresponds to your version of ESP32. Most often it will be
the ESP-WROOM DA Module board

We configured the Arduino development environment to be able to program the
ESP-WROOM DA module.

The installation is now complete, you can now program your ESP32 card
like a classic Arduino board. Examples are available, among others, for Wi-Fi and
Bluetooth in **File → Examples**

### Upload your first program to the ESP32-VROOM-3D card
Now that all the tools are installed, we will be able to create our first program and
send it to the ESP32 card. But before doing our first program, we must ensure that
we can communicate with the card. Normally, ESP32 cards are automatically detected,
there is therefore no need to install additional drivers. Plug your ESP32 board into your
computer via USB (with a USB Micro-B cable), The red LED lights up when powered on
of the ESP32 card.

**The parameter that can be modified in the serial monitor is the transmission speed of the
data**. In general, the typical serial port speed with an Arduino board is **9600 bauds**
(bit/s). But with the ESP32, the commonly used speed is **115200 bauds**. The reason is that
The ESP32 sends debug messages at this speed when it starts up.
The serial port speed may be different from 115200 baud but in this case the messages when
boot of the ESP32 will not be visible and will appear with incomprehensible characters.
Once the speed is changed, you can communicate with the ESP32.

### My first program
Here is a very simple example which allows you to flash the blue LED located on the GPIO2 pin of the
ESP32 Wroom DevKit board.

```
void setup() {
Serial.begin(115200);
Serial.println("Mon premier programme");
pinMode(2,OUTPUT);
}
void loop() {
digitalWrite(2, HIGH); //On allume la led bleue
delay(500); //On attend pendant 500 ms
digitalWrite(2, LOW); //On eteinds la led bleue
delay(500); //On attend pendant 500 ms
}
```
### Compile the program

Once you have typed or copied the example code and saved the code, you will need to compile
the program. The first compilation is quite long because all the source files for the ESP32,
even those not directly used by the program are compiled. Fortunately
only modified files will be compiled at the next compilation.
To do this, simply press the “Check” button. You will then have to click on the arrow next to
to upload the program to the ESP32.
Once the upload is complete, the ESP32 directly executes the program. It is therefore
normal not to see the message “My first program” when you go to the monitor
series.

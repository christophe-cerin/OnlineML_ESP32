
# IDE (Integrated Development Environment)

The integrated development environment (Integrated Development Environment abbreviated IDE) is
a software application that helps programmers develop software code efficiently. A
IDE typically includes a graphical interface to launch the different tools, an editor for
source code, a compiler, a debugger, and often a tool for building
software graphical interfaces

## Arduino IDE

Makers, students, and professionals use the classic Arduino IDE (environment
integrated development) since the birth of Arduino. The Arduino IDE 2.0 is an improvement
classic IDE, with increased performance, improved user interface and
many new features, such as auto-completion, a built-in debugger and
synchronization of sketches with Arduino Cloud.
The new major version of the Arduino IDE is faster and even more powerful! In addition to a
more modern editor and a more responsive interface, it offers auto-completion, navigation
in code and even a live debugger.

There are in fact several methods for programming an ESP32 card, 
but for reasons standardization and to limit the number of software, 
we will use the Arduino IDE tool which is quite easy to use. 
The Arduino card is very easily programmable with this software,
however it will be necessary to make some preparations to be able to program the ESP32,
in particular install the ESP32 card itself.
Download and install the Arduinode IDE from the official [Arduino.cc](https://www.arduino.cc/en/software) Website.

# Preparing the ESP-WROOM-32D microcontroller 

To develop applications for ESP-WROOM-32D, you needed :

- PC loaded with Ubuntu 22.04.1  LTS operating system
- Toolchain to create application for ESP32
- ESP-IDF or Arduino which essentially contains the API for ESP32 and scripts to do
operate toolchain
- A text editor to write programs (Projects) in C for example Eclipse
- The ESP32 board itself and a USB cable to connect it to the PC

![Microcontrôleur ESP32-DevkitC équipé de WROOM-32](https://github.com/christophe-cerin/OnlineML_ESP32/blob/main/ARDUINO/images/carte-esp32-wroom-32d.png)

###### ESP32-DevkitC microcontroller equipped with WROOM-32D 

# Presentation of the ESP-WROOM-32D microcontroller

ESP32-DevKitC on a turnkey basis is an entry-level ESP32 development board
range, it is also a small electronic card, called a microcontroller, easy to take
in hand. On the ESP32 board, the pins are pinned for connection and use
easy. Can be used to evaluate ESP-WROOM-32 modules and ESP32-D0WD chips.
- Flash / PSRAM: 4 MB flash,
- Interface: I/O,
- USB,
- User interface: button, LED
Related products: ESP-WROOM-32

There are a multitude of ESP32 boards with different pin placement. Standard serial ports on
gpio3 (RX) and gpio1 (TX) connections are used in series to communicate with arduino IDE and to be connected to the
CP2102.

## Special operation of certain ESP32 pins

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
(to upload the program to flash memory). In fact, depending on the tension present at the edge of these
pins, the ESP32 will boot either in BOOT mode or in FLASH mode. The strapping pins are the pins
GPIO0, GPIO2, GPIO12 (MTDI) and GPIO15 (MTDO). You can use them, but you just have to be careful
when a logic state (3.3V or 0V) is imposed with an external pullup or pulldown resistor.
- When booting the ESP32, for a short period of time, certain pins quickly change logical states (0V
→ 3.3V). You may have weird bugs with these pins: for example a relay that activates
temporarily. The faulty pins are as follows:
- GPIO 1: Send the ESP32 boot logs via the UART
- GPIO 3: Voltage of 3.3V during boot
- GPIO 5: Sends a PWM signal during boot
- GPIO 14: Sends a PWM signal during boot
- GPIO 15: Sending the ESP32 boot logs via the UART
- The EN pin allows you to control the ignition status of the ESP32 via an external wire. It is connected to the EN button of the
map. When the ESP32 is turned on, it is at 3.3V. If we connect this pin to ground, the ESP32 is turned off. We can use it
when the ESP32 is in a case and you want to be able to turn it on/off with a switch.
- The rest of the GPIO pins have no particular restrictions.

  ![Microcontrôleur ESP32-DevkitC équipé de WROOM-32](https://github.com/christophe-cerin/OnlineML_ESP32/blob/main/ARDUINO/images/carte-ESPWROOM32D.png)

  ###### the GPIO pins of the ESP32-WROOM-32D




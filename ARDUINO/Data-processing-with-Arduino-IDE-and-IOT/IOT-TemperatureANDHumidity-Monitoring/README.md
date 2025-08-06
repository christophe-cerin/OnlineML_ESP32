# IoT Temperature and Humidity Monitoring: An Integrated Solution

**Abstract**

We want to develop a temperature and humidity monitoring system based on the Internet of Things (IoT). To do this, we are looking for an all-in-one platform that will allow us to design, deploy, and host our solution securely and efficiently.

This platform must meet several key criteria:
 - Accessibility and Flexibility: Be accessible via the internet to facilitate the management, modification, and evolution of our projects.
 - Performance and Reliability: Guarantee optimal performance, high availability, and scalability for all deployed instances.
 - Monitoring and Analysis: Provide logging and telemetry tools to allow us to monitor application behavior and respond quickly when needed.
 - Simplified Development (Low-Code): Include Low-Code development features, such as a device manager, dashboard builder, rules engine, workflow engine, alerting system, scheduler, and a multi-tenant architecture.
  - Mobile Compatibility: Be usable on both computers and mobile phones.

**Our Electronic Circuit**

Our prototype is based on three main components:
 - An ESP32-WROOM-32EU microcontroller for data processing.
 - A DHT22 sensor to measure temperature and humidity.
 - A 16x2 LCD screen to display information.

To simplify wiring and save the microcontroller's I/O pins, we can use an I2C conversion module for the LCD. This adapter converts the LCD screen's protocol to an I2C (Inter-Integrated Circuit) interface, a widely used serial communication protocol. This choice allows us to connect the screen with fewer wires, making the assembly cleaner and simpler. Unfortunately, we do not have one at this time.

**Hardware, Wiring, and Configuration**

For this project, you will need the following items:
 - **Required Hardware :**
  - An ESP32 microcontroller
  - A 16x2 LCD screen
  - A potentiometer (to adjust the LCD screen's contrast)
  - A DHT22 sensor (to measure temperature and humidity)
  - 2 resistors (to protect the LCD screen and the LED) and a LED indicator

- **Wiring :**
 - LCD Screen: Connect the LCD screen directly to the ESP32. You will need several pins, particularly for data signals, the register select, and the enable pin. Be sure to follow the specific wiring diagram for your screen.
 - Potentiometer: The potentiometer must be connected to the VCC (power), GND (ground), and the contrast adjustment pin (Vo) of the LCD screen. By turning the potentiometer, you will adjust the screen's contrast for better readability.
 - DHT22 Probe: The DHT22 sensor connects to the ESP32 with three wires: VCC (3.3V power), GND (ground), and a data pin. The data pin must be connected to a digital input pin on the ESP32.
This direct wiring will allow you to control your entire monitoring system. Once the hardware is in place, you can write the necessary code to read data from the DHT22 sensor and display it on the LCD screen.

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


<figure>
   <img alt="Matrice X" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/IOT-Temperature-Humidity-Monitoring/images/circuitEspLcdBreadPot.png"/>
    <figcaption><b>Figure 2:</b> Component and Circuit Diagram</figcaption>
</figure>


<figure>
   <img alt="Matrice X" align="center" width=30% height=30% src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/IOT-Temperature-Humidity-Monitoring/images/lcd16pinsdetailsvertical.png"/>
    <figcaption><b>Figure 2:</b> The LCD screen and 16 pins</figcaption>
</figure>

## Wiring and Required Hardware

Here is a summary table of the necessary hardware and connections for your project.

| Required Hardware | Wiring |
| --- | --- |
| 2 x 1 k Ohm Resistors | Protect the screen and LED from power surges. |
| Potentiometer | The third pin is connected to the V0 / Contrast pin of the LCD screen. |
| DHT22 Sensor | The green wire is connected to pin 15 of the ESP32 board. The red wire is connected to the positive rail of the breadboard, and the black wire to the negative rail. |
| | - LCD terminal pin - connected to the GND pin of the ESP32. |
| | - LCD terminal pin + connected to the 5V pin of the ESP32. |
| | - LCD D7 pin connected to pin 19 of the ESP32. |
| | - LCD D6 pin connected to pin 23 of the ESP32. |
| | - LCD D5 pin connected to pin 18 of the ESP32. |
| 16x2 LCD Screen (16 pins) | - LCD D4 pin connected to pin 5 of the ESP32. |
| | - LCD Enable pin connected to pin 21 of the ESP32. |
| | - LCD R/W pin connected to the GND pin of the ESP32. |  
| | - LCD RS pin connected to pin 22 of the ESP32. |
| | - LCD VCC/VDD pin connected to the 5V pin of the ESP32. | 
| | - LCD VSS pin connected to the GND pin of the ESP32. |
| Microcontroller Unit (MCU) Board | ESP WROOM 32UE. |
| Breadboard | Breadboard |
| One microUSB and USB2 cable | For powering and programming the ESP32. |
| LED | The LED is connected to pin 21 of the ESP32. |

## Arduino IDE Libraries to Install

To make your temperature and humidity monitoring project work, you will need to install the following libraries in your Arduino IDE:

| Library | Function |
| --- | --- |
| DHT22 (by dvarrel), SimpleDHT (by winlin) | These allow you to read data from the temperature and humidity sensor. |
| LiquidCrystal (by Adafruit and Arduino) | Manages the display on the LCD screen. |
| DHT sensor library (by Adafruit) | Another option for interfacing with the temperature and humidity sensor. |
| Adafruit AM2320 (by Adafruit) | A library for another type of temperature and humidity sensor, the AM2320. |

## Guide to Using the Blynk.cloud Platform
To use Blynk.cloud, follow these steps to set up a dashboard and connect your device:

**1. Account and Project Creation**

 - Create an account on the Blynk.cloud platform.
 - Once logged in, create a new Dashboard for your project.

**2. Datastream Configuration**

The Datastream allows you to define the data flows between your device and the dashboard.

 - Create a virtual pin V0 for temperature: Define the variable as a float (decimal) for "Temperature (°C)" with a value range of 0 to 50.
 - Create a virtual pin V1 for humidity: Define the variable as a float (decimal) for "Humidity (%)" with a value range of 0 to 100.
 - Create a virtual pin V2 for an LED: Define the variable as a boolean (true/false) for "LED".
 - Create a virtual pin V3 for a switch: Define the variable as a boolean (true/false) for "SWITCH".

**3. Web Dashboard Configuration**

Repeat the same steps as for the Datastream in the "Web Dashboard" section to display the data on Blynk's web interface.

**4. Device Addition**

 - Add a new device by selecting the ESP32-WROOM-DA-MODULE model.
 - Give your device a name and associate it with the template you previously created in the Dashboard.

**5. ESP32 Connection**

 - Copy your device's authentication information, which looks like this:
 ``` 
#define BLYNK_TEMPLATE_NAME "xxxxx"
#define BLYNK_AUTH_TOKEN "yyyyy"
```
 - Open the Arduino IDE. Go to Examples → Blynk → Boards_Wifi → Esp32_Wifi.
 - Paste the authentication information you just copied into the example code.
 - In your Blynk dashboard, you will find another piece of information to copy :
```    
#define BLYNK_TEMPLATE_ID "zzzzz"
#define BLYNK_TEMPLATE_NAME "xxxxx"
``` 
 - Also paste this information into your Arduino code. You can then adapt the program to your specific needs.
 - The program that we have successfully developed and tested is now ready to be uploaded to your ESP32.


Once the program has been uploaded to the board, you can watch in the IDE's **serial monitor** as the connection is established between your computer and the Blynk IoT Cloud (Figure 1). You'll then be able to track the real-time data directly on the Blynk **Dashboard**.

## Arduino IoT Cloud Guide for Temperature and Humidity Monitoring

This guide explains how to use an **ESP32-WROOM-32UE** microcontroller with a **DHT22** sensor and a **16x2 LCD** screen to monitor temperature and humidity. You'll learn how to use the **Arduino IoT Cloud** platform to view this data on your phone or computer, wherever you are.

### 1. Project Overview

This project will show you how to integrate your sensors with the Arduino IoT Cloud. You'll be able to remotely monitor the temperature and humidity of a location.

### 2. Setting Up Your Project on Arduino IoT Cloud

Once your electronic circuit is assembled, log in to the **Arduino IoT Cloud** platform and follow these steps to configure your project:

**Create a "Thing"**

A **Thing** is a container that groups all the elements of your project, such as devices, Cloud variables, code, and metadata.
 - Name your **Thing**.
 - Create **Cloud variables** for temperature and humidity. These are specific variables that synchronize data between your board and the platform.
 - Associate your board (**ESP32**) with your **Thing**.
 - Configure your board's Wi-Fi settings.

### 3. Device Installation and Configuration

Before associating your device, make sure you have installed the **Arduino Create Agent**. This tool allows the Arduino IoT Cloud to detect boards connected to your computer. Once the agent is installed, you can connect your board and configure its Wi-Fi settings.

### 4. Uploading the Code

 - Go to the Sketch tab to access the online code editor.
 - Copy the code provided in the corresponding section of this guide and paste it into the editor.
 - Click the upload button. The code will be verified, then transferred to your board.

Your device is now operational and sending data to the Arduino IoT Cloud!

### 5. Visualizing Data on a Dashboard
   
To visualize your sensor data, you need to create a dashboard.

 - Go to the tab dedicated to configuring widgets.
 - Add a "Gauge" widget for temperature.
 - Add a "Percentage" widget for humidity.
 - For each widget, give it a name and associate it with the corresponding Cloud variable (Temperature or Humidity).
 - Click "DONE".

Your dashboard is ready and displays the temperature and humidity values captured by your device in real time.


<figure>
   <img alt="Matrice X" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/IOT-Temperature-Humidity-Monitoring/images/connectingBlynkaffichage.jpg" width=50% height=70% />
    <figcaption><b>Figure 1:</b> Image of the Blynk Cloud connection</figcaption>
</figure>

<figure>
   <img alt="Matrice X" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/IOT-Temperature-Humidity-Monitoring/images/iotarduinocloud-esp32LCDwithoutRyverski.png" width=70% height=70% />
    <figcaption><b>Figure 2:</b> Program, Serial Port and Data</figcaption>
</figure>

<figure>
   <img alt="Matrice X" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/IOT-Temperature-Humidity-Monitoring/images/circuitesplcdpc.jpg" width=30% height=30% />
    <figcaption><b>Figure 3:</b> Electronic Circuit</figcaption>
</figure>

<figure>
   <img alt="Matrice X" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/IOT-Temperature-Humidity-Monitoring/images/dashbordIotCloudesp32LCDryverski.png"  width=60% height=60% />
    <figcaption><b>Figure 4:</b> Dashboard on our computer's screen</figcaption>
</figure>


<figure>
   <img alt="Matrice X" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/IOT-Temperature-Humidity-Monitoring/images/dashBoardEspBlynkTempHumTelephone.jpg"  width=30% height=30% />
    <figcaption><b>Figure 4:</b> This figure shows the Arduino IoT Cloud dashboard as it appears on our phone</figcaption>
</figure>

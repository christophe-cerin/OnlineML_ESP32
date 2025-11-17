# IoT Weather Station: Integrating the BME280 into Node-RED via the Arduino Uno R4

## Summary and Objectives

This project aims to build an autonomous weather station using the Arduino Uno R4 WiFi board and the BME280 environmental sensor. The main objective is to retrieve and visualize environmental data (temperature, humidity, atmospheric pressure and sea-level) on a user-friendly web dashboard using the Node-RED platform.
The overall process is divided into two main phases :

1. **Arduino Code :** Reading data from the BME280 and sending it via the USB-C serial port.
1. **Node-RED Flow :** Receiving, parsing the serial data, and displaying it on a user interface (Dashboard).

## Arduino Configuration

The Arduino Uno R4 WiFi board acts as the client, reading sensor data and transmitting it to the host computer.

### Prerequisites and Connections

- **Hardware :** An Arduino Uno R4 WiFi board connected to a laptop via the USB-C port.
- **Required Libraries :** To read the BME280 sensor, the Adafruit Unified Sensor and Adafruit BME280 libraries must be installed via the Arduino IDE's Library Manager.
- **I2C Wiring :** The BME280 communicates with the Arduino using the I2C (Inter-Integrated Circuit) protocol.

| BME280 | Arduino Uno R4 | Note | 
| :--- | :--- | :--- | 
| VCC | 5V or 3.3V | Depends on the BME280 module |
| GND | GND | Ground |
| SCL | A5 (or dedicated SCL) | Clock (Blue Wire) |
| SDA | A4 (or dedicated SDA) | Data (Green Wire) | 


### Arduino Code Functionality

The Arduino program is responsible for sensor initialization and data transmission.
- **Initialization (setup) :** Serial communication is initialized at 9600 bauds (Serial.begin(9600)). The BME280 is initialized, typically attempting the I2C address 0x77 (or 0x76 depending on the module).
- **Main Loop (loop) :**
  - The values for Temperature (bme.readTemperature()), Humidity (bme.readHumidity()), Pressure (bme.readPressure()), and Altitude (bme.readAltitude()) are read. Pressure is converted to hPa by dividing the output by 100.0F.
  - The four values are sent over the serial port in a comma-separated (CSV) format, specifically T,H,P,A.
  - A 1.5 second delay (delay(1500)) is introduced between readings.
Serial Data Format Example: Data is sent as a single string, such as: 25.98,38.67,1005.44,65.26 (Temperature, Humidity, Pressure, Altitude).

## Node-RED Configuration

Node-RED is a low-code programming platform based on Node.js, ideal for IoT and creating visual APIs, allowing easy connection of hardware, APIs, and services.

### Node-RED Prerequisites

- **Installation :** Node-RED must be installed and running (the editing interface is usually accessible at http://127.0.0.1:1880).
- **Dashboard Module :** The Node Dashboard (node-red-dashboard) node must be installed to create the gauges and charts.
- **Serial Conflict :** It is critical to ensure the Arduino IDE Serial Monitor is closed before starting Node-RED, as both cannot use the serial port simultaneously.
    
### The Node-RED Flow

The flow is a set of nodes connected by wires, representing the program logic26. It consists of four main node types:
- **Serial In Node :** Receives data from the Arduino.
  - Configuration : The serial port (e.g., /dev/ttyACM0) and the baud rate (9600) must match the Arduino configuration.
- **Function Node :** Parses and structures the data.
  - This node, named Parse BME280 Data, receives the CSV string (T,H,P,A) via msg.payload.
  - It uses the JavaScript function split(',') to separate the values.
  - It then creates a JavaScript object (JSON) by converting each string to a number (parseFloat) and stores it in msg.payload for subsequent nodes:
        
 ```json
                                         msg.payload={temp,hum,press,alt}
 ``` 

- **Change Nodes :** Separates the JSON object properties.
  - Four Change nodes are used (one for each measurement: Temperature, Humidity, Pressure, Altitude).
  - Each node takes a property from the object (e.g., msg.payload.temperature) and moves it to the message root (msg.payload) to easily feed the gauges and charts.
- **Dashboard Nodes (Gauge/Text/Chart) :** Display the data on the web interface36.
  - Gauges are used for Temperature (Range: 0 to $40^{\circ}C$) and Humidity (Range: 0 to 100%).
  - Text displays are used for Pressure (in hPa) and Altitude (in m).
  - Line charts are used to visualize the history of the four measurements.

## Analysis and IoT Potential

### The Role of Node-RED

Node-RED is the cornerstone of this IoT solution. It transforms a simple USB/Serial data transmission into a complete web application (the Dashboard).
- **Immediate Visualization :** The Dashboard allows real-time data visualization on the host computer (http://127.0.0.1:1880/ui) and on mobile devices connected to the same network.
- **Low Complexity :** The approach uses graphical programming by connecting nodes, limiting the need to write complex code (except for the parsing Function node).

### Evolution and Improvements

This project serves as a solid foundation but can be easily extended thanks to Node-RED's capabilities.
- **WiFi Autonomy :** The use of the Arduino Uno R4 WiFi paves the way for sending data directly over the network (via MQTT or HTTP), making the station completely autonomous (without a USB connection to the host computer).
- **Data Storage :** The Node-RED flow can be easily extended to connect the Change nodes to a database (such as MySQL or MongoDB) to archive the weather readings.
- **Alerts and Automation :** Additional nodes could be added to send notifications (email, SMS, Telegram) if a value exceeds predefined thresholds (e.g., temperature > $30^{\circ}C$), enabling control and command applications.

### Illustrations 

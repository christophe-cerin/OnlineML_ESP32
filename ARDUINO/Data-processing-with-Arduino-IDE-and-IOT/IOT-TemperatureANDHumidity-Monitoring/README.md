# IoT Temperature and Humidity Monitoring: An Integrated Solution

**Abstract**

We want to develop a temperature and humidity monitoring system based on the Internet of Things (IoT). To do this, we are looking for an all-in-one platform that will allow us to design, deploy, and host our solution securely and efficiently.
This platform must meet several key criteria:
    • Accessibility and Flexibility: Be accessible via the internet to facilitate the management, modification, and evolution of our projects.
    • Performance and Reliability: Guarantee optimal performance, high availability, and scalability for all deployed instances.
    • Monitoring and Analysis: Provide logging and telemetry tools to allow us to monitor application behavior and respond quickly when needed.
    • Simplified Development (Low-Code): Include Low-Code development features, such as a device manager, dashboard builder, rules engine, workflow engine, alerting system, scheduler, and a multi-tenant architecture.
    • Mobile Compatibility: Be usable on both computers and mobile phones.

Our Electronic Circuit
Our prototype is based on three main components:
    • An ESP32-WROOM-32EU microcontroller for data processing.
    • A DHT22 sensor to measure temperature and humidity.
    • A 16x2 LCD screen to display information.
To simplify wiring and save the microcontroller's I/O pins, we can use an I2C conversion module for the LCD. This adapter converts the LCD screen's protocol to an I2C (Inter-Integrated Circuit) interface, a widely used serial communication protocol. This choice allows us to connect the screen with fewer wires, making the assembly cleaner and simpler. Unfortunately, we do not have one at this time.****

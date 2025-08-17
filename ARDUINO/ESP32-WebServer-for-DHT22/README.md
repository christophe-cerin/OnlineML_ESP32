## ESP32 Web Server for DHT22 Temperature and Humidity Sensor

The ESP32 is an affordable, energy-efficient chip with built-in Wi-Fi and Bluetooth. Using the Arduino integrated development environment, we can turn it into a web server to display real-time temperature and humidity data from a DHT22 sensor. The ESP32 will read the sensor data and show it on a web page.

### How the System Works

This project sets up a web server on the ESP32 that will:
 - Host the necessary HTML and CSS files for the web page.
 - Send these files to any client (like your web browser) that requests them.
 - Display real-time data from the DHT22 sensor on the web page.
 - Be configured to control the ESP32's GPIO pins.

To access the server, the ESP32 must be connected to the same Wi-Fi network as your device (phone or computer). The web server is simply the location where the page files are stored, and it delivers them to clients upon request.

### Communication Between a Web Client and a Server

For a client (your web browser or an application) and a web server (in this case, your ESP32) to communicate, they use a common language: the HTTP protocol.
When you want to access a web page, your browser sends an HTTP GET request to the server. The server receives, processes the request, and then sends back the web page files. If the page is unavailable for any reason, the server sends an error message, like the well-known 404 error. A single server can handle requests from multiple clients at once.

### The Different Modes of the ESP32

The ESP32 is very versatile and can act as a web server in two different ways.

 - 1. Station Mode (STA)

In this mode, your ESP32 connects to an existing Wi-Fi network (like the one from your internet router). This is the most common mode. The ESP32 gets an IP address from your router and becomes accessible to all other devices on the same network. They can then connect to the web server by entering the ESP32's IP address into their browser.

 - 2. Access Point Mode (AP)
      
In this mode, the ESP32 creates its own Wi-Fi network. It doesn't need to connect to another network. Other devices can connect directly to this network, just as they would to a home router. Once connected, they can access the ESP32's web server using its IP address. This mode is very useful for on-site use where an existing Wi-Fi network isn't available.

### Using the ESP32 as a Web Server in "Station" Mode

Station mode (or STA) allows your ESP32 board to connect to your Wi-Fi network, just like a phone or computer. Once connected, the ESP32 can act as a web server, serving web pages to all devices on the same network.
To access it, simply know the IP address assigned to your ESP32 by your router and type it into your web browser's address bar. This is the ideal mode for home or office projects.

### How to find the IP address of your ESP32 server

Your ESP32 is a server, and just like a website on the internet, it needs a unique address to be accessible. This is the role of the IP address.
When your ESP32 connects to a Wi-Fi network, it is assigned an IP address by that network. This address is what allows other devices, like your computer or phone, to find it and access the web server you created. Simply type this IP address into your browser's address bar to connect to it.

<figure>
   <img alt="Matrice X" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/ESP32-WebServer-for-DHT22/images/ArduinowebAsync.png"/>
    <figcaption><b>Figure 2:</b> Electronic Circuit</figcaption>
</figure>

### How to Access the ESP32 Web Page

To view the web page hosted on your ESP32, you need its IP address. This unique address allows your browser to find it on the network.
The process is simple:

 - 1. Obtain the IP address: Your program will display the ESP32's IP address.
 - 2. Enter the address: Type this address into your web browser's search bar.
 - 3. Display the page: The browser will send a request to the ESP32 and display the web page stored there.

This page will display the information you programmed, such as temperature and humidity data from the DHT22 sensor.

<figure>
   <img alt="Matrice X" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/ESP32-WebServer-for-DHT22/images/httpwebAsync.png"/>
    <figcaption><b>Figure 2:</b> Electronic Circuit</figcaption>
</figure>

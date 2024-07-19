
import ubinascii
import math
import random

# WiFi and MQTT credentials
SSID = 'wifi-test'
PASSWORD = 'Sunny789'
MQTT_BROKER = '130.190.72.193'
MQTT_PORT = 1883
MQTT_TOPIC1 = 'co2/celsius'
MQTT_TOPIC2 = 'final/final'

# WiFi connection
def connect_wifi(ssid, password):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print('Connecting to network...')
        wlan.connect(ssid, password)
        while not wlan.isconnected():
            time.sleep(1)
    print('Network config:', wlan.ifconfig())

# Point class and helper functions
class Point:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
        self.cluster = -1
        self.minDist = float('inf')

    def distance(self, p):
        return (self.x - p.x) ** 2 + (self.y - p.y) ** 2

def copy_points(points):
    return [Point(point.x, point.y) for point in points]

def print_points(points):
    for i, point in enumerate(points):
        print(f"{i} -> {point.x}, {point.y}")
    print("\n")
def less_points(lhs, rhs):
    """Custom less comparator"""
    return (lhs.x < rhs.x) or ((lhs.x == rhs.x) and (lhs.y < rhs.y))

def less_equal_points(lhs, rhs):
    """Custom less or equal comparator"""
    return (lhs.x <= rhs.x) or ((lhs.x == rhs.x) and (lhs.y <= rhs.y))

def greater_points(lhs, rhs):
    """Custom greater comparator"""
    return (lhs.x > rhs.x) or ((lhs.x == rhs.x) and (lhs.y > rhs.y))

def greater_equal_points(lhs, rhs):
    """Custom greater or equal comparator"""
    return (lhs.x >= rhs.x) or ((lhs.x == rhs.x) and (lhs.y >= rhs.y))

def equal_points(lhs, rhs):
    """Custom equal comparator"""
    return (lhs.x == rhs.x) and (lhs.y == rhs.y)

def printarr(arr):
    """A utility function to print an array"""
    for i, point in enumerate(arr):
        print(f"{i} -> {point.x}, {point.y}")
    print()

def clear_buffer(points, mycount):
    arr_copy = points[:]
    
    taille = int(math.sqrt(mycount))
    mysize = mycount - 1
    
    i = taille - 1
    while i < mysize:
        found = False
        for v in points:
            if equal_points(v, arr_copy[i]):
                found = True
                break
        
        if not found:
            points = [v for v in points if not equal_points(v, arr_copy[i - 1])]
        else:
            points = [v for v in points if not equal_points(v, arr_copy[i])]
        
        i += taille
    
    return points

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and (arr[j].x > key.x or (arr[j].x == key.x and arr[j].y > key.y)):
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def quicksort(arr, l, r):
    if r <= l:
        return
    if (r - l + 1) <= 1024:
        insertion_sort(arr[l:r+1])
        return

    i, j = partition(arr, l, r)
    quicksort(arr, l, j)
    quicksort(arr, i, r)

def partition(arr, l, r):
    i = l - 1
    j = r
    p = l - 1
    q = r
    v = arr[r]

    while True:
        while arr[i + 1].x < v.x or (arr[i + 1].x == v.x and arr[i + 1].y < v.y):
            i += 1
        while arr[j - 1].x > v.x or (arr[j - 1].x == v.x and arr[j - 1].y > v.y):
            j -= 1
            if j == l:
                break
        if i >= j:
            break
        arr[i], arr[j] = arr[j], arr[i]
        if arr[i].x == v.x and arr[i].y == v.y:
            p += 1
            arr[p], arr[i] = arr[i], arr[p]
        if arr[j].x == v.x and arr[j].y == v.y:
            q -= 1
            arr[q], arr[j] = arr[j], arr[q]

    arr[i], arr[r] = arr[r], arr[i]

    j = i - 1
    for k in range(l, p + 1):
        arr[k], arr[j] = arr[j], arr[k]
        j -= 1

    i = i + 1
    for k in range(r - 1, q, -1):
        arr[k], arr[i] = arr[i], arr[k]
        i += 1

    return i, j

def kMeansClustering(points, epochs, k):
    n = len(points)
    # Randomly initialize centroids
    centroids = [points[random.randint(0, n - 1)] for _ in range(k)]

    for _ in range(epochs):
        # For each centroid, compute distance from centroid to each point
        # and update point's cluster if necessary
        for clusterId, centroid in enumerate(centroids):
            for point in points:
                dist = centroid.distance(point)
                if dist < point.minDist:
                    point.minDist = dist
                    point.cluster = clusterId

        # Create lists to keep track of data needed to compute means
        nPoints = [0] * k
        sumX = [0.0] * k
        sumY = [0.0] * k

        # Iterate over points to append data to centroids
        for point in points:
            clusterId = point.cluster
            nPoints[clusterId] += 1
            sumX[clusterId] += point.x
            sumY[clusterId] += point.y
            point.minDist = float('inf')  # reset distance

        # Compute the new centroids
        for clusterId in range(k):
            if nPoints[clusterId] > 0:  # prevent division by zero
                centroids[clusterId].x = sumX[clusterId] / nPoints[clusterId]
                centroids[clusterId].y = sumY[clusterId] / nPoints[clusterId]

    print("The data points and the Centroids with their cluster-id :")
    for point in points:
        print(f"[[{point.x}, {point.y}], {point.cluster}],")

# Example usage
#points = [Point(random.random(), random.random()) for _ in range(4)]
#kMeansClustering(points, 10, 4)   

def safe_float_conversion(value):
    try:
        return  float(value.strip())
    except ValueError:
        return 0  # or handle it in a way that suits your needs
    
# MQTT message handling
mycount = 0
my_round = 1
arr = []
points = []

def on_message(topic, msg): 
    global mycount, my_round, arr, points
    msg_str = msg.decode('utf-8')

    if msg_str == "0.0,0.0":
        print("End of the DATA reception")
        return
    
    data = msg_str.split(",")
    if len(data) != 2:
        print("Invalid data format")
        return
    
    x = safe_float_conversion(data[0])
    y = safe_float_conversion(data[1])
    myPoint = Point(x, y)  # Create a Point object
    arr.append(myPoint)
    mycount += 1

    if mycount == 128:  # Adjust the count for testing purposes
        print(f"==========\n===== Calculation Turn Number = {my_round} with 1024 vector points and with an iteration of 128 points=====\n==========")
        my_round += 1
        points.extend(arr)
        
        
        kMeansClustering(points, 128, 4)  # Ensure the clustering function is called with correct parameters
        
        print("Displaying a set of 128 data points after quicksorting")
        quicksort(points, 0, mycount - 1)
        printarr(points)

        # Clear buffer points
        points = clear_buffer(points, mycount)
        printarr(points)
        
        arr = []
        mycount = 0
            
#-------- Getting MQTT Data --------
# Connection of MqTT to collect real time data
import network
from umqtt.simple import MQTTClient

# WiFi connection details
WIFI_SSID = 'wifi-test'
WIFI_PASSWORD = 'Sunny789'

# MQTT broker details
MQTT_BROKER = '130.190.72.193'  # Change this to the IP address or hostname of your MQTT broker
MQTT_PORT = 1883
MQTT_USER = 'root'
MQTT_PASSWORD = 'Sunny789'

MQTT_TOPIC1 = 'co2/celsius'
MQTT_TOPIC2 = 'final/final'

def connect_wifi():
    sta_if = network.WLAN(network.STA_IF)
    if not sta_if.isconnected():
        print('Connecting to WiFi...')
        sta_if.active(True)
        sta_if.connect(WIFI_SSID, WIFI_PASSWORD)
        while not sta_if.isconnected():
            pass
    print('WiFi connected:', sta_if.ifconfig())


def main():
    connect_wifi()
    client = MQTTClient("esp32", MQTT_BROKER, MQTT_PORT, MQTT_USER, MQTT_PASSWORD)
    client.set_callback(on_message)
    client.connect()
    client.subscribe(MQTT_TOPIC2)
    print('Connected to MQTT broker')

    try:
        while True:
            client.wait_msg()
    finally:
        client.disconnect()
        print('Disconnected from MQTT broker')


if __name__ == '__main__':
    main()

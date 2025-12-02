import serial, time

# Open COM11 at 115200 baud
ser = serial.Serial('COM11', 115200, timeout=1)
time.sleep(2)   # wait for Arduino reset

# Tell Arduino to start streaming
ser.write(b'S\n')

# Read 20 lines
for i in range(100):
    line = ser.readline().decode(errors='ignore').strip()
    print(line)

ser.close()

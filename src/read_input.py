import serial

import re



print("running..")

port = '/dev/ttyUSB0' #the port the arduino is on.  if things aren't working, this may have changed.. itdefaults back to 0 after restarting.  or type 'ls /dev/tty*' and find the USB '#'



ser = serial.Serial(port, 9600,timeout= None)

if ser.isOpen():

    print("Connected!\nWaiting for data..")

while ser.isOpen():
    data = ser.readline().decode('utf-8') #read the buffer.  clears it automatically after reading

    if len(data) > 0:
        print('recieved:', data) #if something was there, print it!
        #You can extract the time with this:
        time = [int(s) for s in data if s.isdigit()]
        time = int(re.findall(r'\d+',data)[0])
        print(time)

ser.close()

print("Disconnected")

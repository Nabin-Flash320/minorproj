
import io
import struct
import socket
import time
import picamera
import sys
from smbus import SMBus
from multiprocessing import Process
       

    
def to_from_arduino(ip_args):
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(ip_args)
    client_socket.sendall(b'ard')
    
    bus = SMBus(1)
    i = 0
    while True:
        vals = bus.read_i2c_block_data(0x9, 6)
        valstr = str()
        print(vals)
        for val in vals:
            if val == 255:
                break
            else:
                valstr += ' '+str(val)
        client_socket.send(bytes(valstr, 'utf8'))
        flag  = client_socket.recv(1024).decode()
        print('Status flag => ', flag)
        print()
        time.sleep(3)
        bus.write_byte(0x9, 2)
        i += 1
    bus.close()
    client_socket.close()





def sendvid(ip_args):

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(ip_args)
    connection = client_socket.makefile('wb')
    client_socket.sendall(b'vid')
    try:
        camera = picamera.PiCamera()
        camera.resolution = (640, 480)
        print('Starting camera.....')
        time.sleep(2)
        stream = io.BytesIO()
        for foo in camera.capture_continuous(stream, 'jpeg'):
            connection.write(struct.pack('<L', stream.tell()))
            connection.flush()
            stream.seek(0)
            connection.write(stream.read())
            stream.seek(0)
            stream.truncate()
        
        

        camera.close()
    finally:
        connection.close()
        client_socket.close()

if __name__ == '__main__':
    
    ip_argus = (sys.argv[1], int(sys.argv[2]))
    proc_1 = Process(target=sendvid, args=(ip_argus, ))
    proc_2 = Process(target=to_from_arduino, args=(ip_argus, ))
    proc_1.start()
    proc_2.start()
    proc_1.join()
    proc_2.join()

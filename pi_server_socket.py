from __future__ import division
import time
import datetime
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from DNModel import net as Darknet
from img_process import inp_to_image, custom_resize
import pandas as pd
import random 
import pickle as pkl
import io
import socket
import struct
from PIL import Image
import sys
from multiprocessing import Process
from _thread import *
from pathlib import Path
import csv


confidence = 0.5
nms_thesh = 0.4
start = 0
CUDA = torch.cuda.is_available()
num_classes = 80

bbox_attrs = 5 + num_classes

print("Loading network")
model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
print("Network loaded")
classes = load_classes('data/coco.names')
colors = pkl.load(open("pallete", "rb"))
model.DNInfo["height"] = 128
inp_dim = int(model.DNInfo["height"])
model.eval()
lbls = 0

pwd = str(Path.cwd())
date_time = datetime.datetime.now()
csv_data_file = pwd + '/rescue_data/rescue_data' + date_time.strftime("_%Y_%m_%d-%H:%M:%S")+'.csv'
Path.touch(Path(csv_data_file))





def prepare_input(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    Perform tranpose and return Tensor
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (custom_resize(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img, con):
	global lbls
	c1 = tuple(x[1:3].int())
	c2 = tuple(x[3:5].int())
	cls = int(x[-1])
	label = "{0}".format(classes[cls])
	#(width, height) = (640, 480)
	if (c1[0], c1[1]) >= (10, 10):
		if label == 'person':
			lbls = 1
		else:
			lbls = 0
		color = random.choice(colors)
		cv2.rectangle(img, c1, c2,color, 1)
	    
		t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
		c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
		if c2[0] == 0:
			return img
		cv2.rectangle(img, c1, c2, color, -1)
		cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);     


	return img


def getvidanddetect(connect):
	connection = connect.makefile('b')
	try:
		img = None
		while True:
		    image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
		    if not image_len:
		        break
		    image_stream = io.BytesIO()
		    image_stream.write(connection.read(image_len))
		    image_stream.seek(0)
		    image = Image.open(image_stream)
		    im = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

		    
		    #Detection phase
		    img, orig_im, dim = prepare_input(im, inp_dim)
		    im_dim = torch.FloatTensor(dim).repeat(1,2) 
		    with torch.no_grad():   
		        output = model(Variable(img), CUDA)
		    output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
		    if type(output) == int:
		    	 cv2.imshow("frame", orig_im)
		    	 key = cv2.waitKey(1)
		    	 if key & 0xFF == ord('x'):
		            break
		    	 continue
		    im_dim = im_dim.repeat(output.size(0), 1)
		    scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
		    
		    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
		    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
		    
		    output[:,1:5] /= scaling_factor

		    for i in range(output.shape[0]):
		        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
		        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
		  
		    list(map(lambda x: write(x, orig_im, connect), output))
		    cv2.imshow('Video',orig_im)
		    if cv2.waitKey(1) & 0xFF == ord('q'):
		        break
		   
		cv2.destroyAllWindows()
	except BrokenPipeError as bpe:
		print("From getvidanddetect==> ", bpe)
	finally:
		connection.close()
        

def get_arduino(connect):
	global lbls
	lbl = lbls
	con = connect
	i = 0
	try:
		data_dict_list = list()
		while True:
			data = str(con.recv(39))
			dats = data.split(' ')
			sensor_list = ['time', 'temp', 'co2', 'ldr', 'u1', 'u2', 'u3', 'gx', 'gy', 'gz', 'ax', 'ay', 'az', 'pir']
			data_dict = {}
			now = datetime.datetime.now()
			now = now.strftime("%H:%M:%S")
			data_dict[sensor_list[0]] = now
			j = 1
			for dat in dats[1:]:
				if "'" in dat:
					l = dat.split("'")
					data_dict[sensor_list[j]] = chr(int(l[0]))
				else:
					data_dict[sensor_list[j]] = dat
				j += 1
			print('Data {0} is {1}'.format(i, data_dict))
			data_dict_list.append(data_dict)
			con.send(str(lbls).encode('utf-8'))
			i += 1	
			try:
			    with open(csv_data_file, 'w+') as csvfile:
			        writer = csv.DictWriter(csvfile, fieldnames=sensor_list)
			        writer.writeheader()
			        for d in data_dict_list:
			            writer.writerow(d)
			except IOError:
			    print("I/O error")
	except BrokenPipeError as bpe:
		print("From get_arduino function==> ", bpe)
	


def mul_thread(connection):
	connection = connection
	while True:
		choice = str(connection.recv(1024))
		break
	if choice == 'b\'ard\'':
		get_arduino(connection)
		
	elif choice == 'b\'vid\'':
		getvidanddetect(connection)
		
	else:
		return



if __name__ == '__main__':
    
    
    """Connecting to the client(RPI)"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
    	server_socket.bind((sys.argv[1], int(sys.argv[2])))
    except socket.error as e:
    	print(e)
    print('Listening....')
    server_socket.listen(5)
    while True:
    	connection, _ = server_socket.accept()
    	start_new_thread(mul_thread, (connection, ))

    
    connection.close()
    server_socket.close()
   

    
    


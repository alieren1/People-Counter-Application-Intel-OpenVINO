"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt
from accuracy import model_accuracy

from argparse import ArgumentParser
from inference import Network


# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def preprocess(frame, net_input_shape):
    p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose((2,0,1))
    p_frame = p_frame.reshape(1, *p_frame.shape)
    return p_frame

def time_conv(t):
    return time.strftime("%H:%M:%S", time.gmtime(t))

def handle_input(input_stream):
    
    image = False
    cam = False
    
    # Input is an image
    if input_stream.endswith('.jpg') or input_stream.endswith('.png') or input_stream.endswith('.bmp'):
        image = True
        
    # Input is a webcam
    elif input_stream == 'CAM':
        cam = True
        
    # Input is a not a video    
    elif not input_stream.endswith('.mp4'):
        log.error('Please enter a valid input!')
        
    return image, cam

def draw_boxes(frame, output, prob_threshold, width, height):
    
    counter = 0
    for box in output[0][0]:
        confidence = box[2]
        
        # Check if confidence is bigger than the threshold and if a person is detected
        if confidence >= prob_threshold and int(box[1]) == 1:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 3)
            counter += 1
                
    return frame, counter

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    request_id = 0
    last_count = 0
    total_count = 0
    duration = 0
    starting_time = 0
    ending_time = 0
    is_detected = False
    count = []
    detection = []
    frame_count = 0
    
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    image, cam = handle_input(args.input)
    if cam:
        args.input = 0
    
    capture = cv2.VideoCapture(args.input)
    capture.open(args.input)
    
    width = int(capture.get(3))
    height = int(capture.get(4))

    ### TODO: Loop until stream is over ###
    while capture.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = capture.read()
        if not flag:
            break
        
        pressed_key = cv2.waitKey(60)
        frame_count += 1

        ### TODO: Pre-process the image as needed ###
        p_frame = preprocess(frame, net_input_shape)

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame, request_id)

        ### TODO: Wait for the result ###
        if infer_network.wait(request_id) == 0:

            ### TODO: Get the results of the inference request ###
            output = infer_network.get_output(request_id)

            ### TODO: Extract any desired stats from the results ###
            out_frame, current_count = draw_boxes(frame, output, prob_threshold, width, height)
            count.append(current_count)
            detection.append(current_count)
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client.publish("person", json.dumps({"count": current_count}))
            
            if current_count > last_count and is_detected == False:
                starting_time = frame_count 
                total_count = total_count + current_count - last_count
                is_detected = True
                
                client.publish("person", json.dumps({"total": total_count}))
            
            if current_count == 0:
   
                if (is_detected and all(x == 0 for x in count[-5:])):
                    is_detected = False 

                    if(count[-6] == 1):
                    
                        ending_time = frame_count - starting_time - 5
                        duration = int((ending_time)/24)
                        client.publish("person/duration", json.dumps({"duration": duration}))
                    
                    else:
                        pass
                
                del count[:-6] 
                last_count = current_count

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush()
        
        if pressed_key == 27:
            break

        ### TODO: Write an output image if `single_image_mode` ###
        if image:
            cv2.putText(out_frame, "current_count: {}".format(current_count), (10, height - ((1 * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imwrite('output_image.jpg', out_frame)
        
    # Print accuracy
    log.warning("Accuracy: {:.2f}%".format(model_accuracy(detection)))  
        
    capture.release()
    cv2.destroyAllWindows()
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    start = time.time()
    infer_on_stream(args, client)
    end = time.time()
    inference_time = time_conv(end - start)
    
    # Print total inference time
    log.warning("Inference time: {}".format(inference_time))


if __name__ == '__main__':
    main()

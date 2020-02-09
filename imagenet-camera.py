#!/usr/bin/python
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils
from multiprocessing import Queue,Process

import argparse
import sys
import json
import os
import time
import threading

from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.parse import quote_plus
from playsound import playsound


API_KEY = 'xxxxxx'
SECRET_KEY = 'xxxxxx'

# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage())

parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")


# 发音人选择, 基础音库：0为度小美，1为度小宇，3为度逍遥，4为度丫丫，
# 精品音库：5为度小娇，103为度米朵，106为度博文，110为度小童，111为度小萌，默认为度小美 
PER = 4
# 语速，取值0-15，默认为5中语速
SPD = 5
# 音调，取值0-15，默认为5中语调
PIT = 5
# 音量，取值0-9，默认为5中音量
VOL = 5
# 下载的文件格式, 3：mp3(default) 4： pcm-16k 5： pcm-8k 6. wav
AUE = 3

FORMATS = {3: "mp3", 4: "pcm", 5: "pcm", 6: "wav"}
FORMAT = FORMATS[AUE]

CUID = "123456PYTHON"

TTS_URL = 'http://tsn.baidu.com/text2audio'


class DemoError(Exception):
    pass


"""  TOKEN start """

TOKEN_URL = 'http://openapi.baidu.com/oauth/2.0/token'
SCOPE = 'audio_tts_post'  # 有此scope表示有tts能力，没有请在网页里勾选
IS_PY3=True

def fetch_token():
    print("fetch token begin")
    params = {'grant_type': 'client_credentials',
              'client_id': API_KEY,
              'client_secret': SECRET_KEY}
    post_data = urlencode(params)
    if (IS_PY3):
        post_data = post_data.encode('utf-8')
    req = Request(TOKEN_URL, post_data)
    try:
        f = urlopen(req, timeout=5)
        result_str = f.read()
    except URLError as err:
        print('token http response http code : ' + str(err.code))
        result_str = err.read()
    if (IS_PY3):
        result_str = result_str.decode()

    # print(result_str)
    result = json.loads(result_str)
    # print(result)
    if ('access_token' in result.keys() and 'scope' in result.keys()):
        if not SCOPE in result['scope'].split(' '):
            raise DemoError('scope is not correct')
        # print('SUCCESS WITH TOKEN: %s ; EXPIRES IN SECONDS: %s' % (result['access_token'], result['expires_in']))
        return result['access_token']
    else:
        raise DemoError('MAYBE API_KEY or SECRET_KEY not correct: access_token or scope not found in token response')

PATH = './sound/'
def text2sound(text):
	if os.path.exists(PATH+text+'.mp3'):
		playsound(PATH+text+'.mp3')
		return
	token = fetch_token()
	tex = quote_plus(text)
	params = {'tok': token, 'tex': tex, 'per': PER, 'spd': SPD, 'pit': PIT, 'vol': VOL, 'aue': AUE, 'cuid': CUID,'lan': 'zh', 'ctp': 1}
	data = urlencode(params)
	req = Request(TTS_URL, data.encode('utf-8'))
	has_error = False
	try:
		f = urlopen(req)
		result_str = f.read()
		headers = dict((name.lower(), value) for name, value in f.headers.items())
		has_error = ('content-type' not in headers.keys() or headers['content-type'].find('audio/') < 0)
	except  URLError as err:
		# print('asr http response http code : ' + str(err.code))
		result_str = err.read()
		has_error = True
	save_file = "error.txt" if has_error else PATH + text + '.' + FORMAT
	with open(save_file, 'wb') as of:
		of.write(result_str)
	playsound(save_file)
	if has_error:
		result_str = str(result_str, 'utf-8')

class Producer(threading.Thread):
	def __init__(self, t_name, queue):
		threading.Thread.__init__(self, name=t_name)
		self.data = queue
	def run(self):
		while True:
			self.data.put(TEXT)


# Consumer thread

class Consumer(threading.Thread):
	def __init__(self, t_name, queue):
		threading.Thread.__init__(self, name=t_name)
		self.data = queue
	def run(self):
		while True:
			val = self.data.get()
			text2sound(val)

ccc_start = False
queue = Queue(1)
consumer = Consumer('Con.', queue)
producer = Producer('Pro.', queue)

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the recognition network
net = jetson.inference.imageNet(opt.network, sys.argv)

# create the camera and display
font = jetson.utils.cudaFont()
camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)
display = jetson.utils.glDisplay()

# process frames until user exits
while display.IsOpen():
	# capture the image
	img, width, height = camera.CaptureRGBA()

	# classify the image
	class_idx, confidence = net.Classify(img, width, height)

	# find the object description
	class_desc = net.GetClassDesc(class_idx)
	TEXT = class_desc
	if not ccc_start:
		producer.start()
		consumer.start()
		ccc_start = True
	# text2sound(class_desc)

	# overlay the result on the image	
	font.OverlayText(img, width, height, "{:05.2f}% {:s}".format(confidence * 100, class_desc), 5, 5, font.White, font.Gray40)
	
	# render the image
	display.RenderOnce(img, width, height)

	# update the title bar
	display.SetTitle("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))
	time.sleep(0.04)
	# print out performance info
	# net.PrintProfilerTimes()
producer.join() #may not stop
consumer.join()


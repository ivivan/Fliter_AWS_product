t = []
import time
from datetime import datetime, timedelta
import boto3
import botocore
import botocore.vendored.requests as requests
import math
from botocore.vendored.requests.exceptions import ConnectionError
import json
import io
from datetime import datetime
import logging 
from eagleFilter import eagleFilter as eagle

def get(node, startTime, endTime):
    HISTORIC_API = 'https://api.eagle.io/api/v1/historic'
    headers = {'X-Api-Key':'GBBbwpSHH54zF58e7Xwp25zFUZ8xJ5c3TxHUff1B'}
    data = []

    params = {'startTime':startTime.strftime('%Y-%m-%dT%H:%M:%SZ'), \
                'endTime':endTime.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'timeFormat' : 'YYYY-MM-DDTHH:mm:ss',
                'timezone' : 'Etc/UTC',
                'limit': 10000, 
                'params':node }
    try:
        hist_values = requests.get(HISTORIC_API, headers=headers, params=params).json()

    except ConnectionError as e :
        print (e)
    return hist_values

for j in range(0,3):
    ea = eagle()
    start = time.clock()
    node =  "5c3578fc1bbcf10f7880ca62"
    source_metadata = ea.getLocationMetadata(node)

    start_time =  source_metadata['currentTime'] - timedelta(days=10)
    finish_time = source_metadata['currentTime']

    # HISTORIC_API = 'https://api.eagle.io/api/v1/historic'
    # headers = {'X-Api-Key':'GBBbwpSHH54zF58e7Xwp25zFUZ8xJ5c3TxHUff1B'}

    # params = {'startTime':start_time.strftime('%Y-%m-%dT%H:%M:%SZ'), \
    #             'endTime':finish_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
    #             'timeFormat' : 'YYYY-MM-DDTHH:mm:ss',
    #             'timezone' : 'Etc/UTC',
    #             'limit': 10000, 
    #             'params':node }

    start = time.clock()
    i = 0
    while (i<5):
        hist_values = requests.get(HISTORIC_API, headers=headers, params=params).json()
        #ea.getData(node, start_time, finish_time)
        i+=1
    fin = time.clock()
    t.append(fin-start)

print("Time: %f from %s" % (sum(t)/len(t), t))

# t = []
# import time
    
# from eagleFilter import eagleFilter as eagle
# from datetime import datetime, timedelta

# ea = eagle()

# for j in range(0,3):
    
#     start = time.clock()
#     node =  "5c3578fc1bbcf10f7880ca62"
#     source_metadata = ea.getLocationMetadata(node)

#     start_time =  source_metadata['currentTime'] - timedelta(days=10)
#     finish_time = source_metadata['currentTime']

#     start = time.clock()
#     i = 0
#     while (i<5):
#         ea.getData(node, start_time, finish_time)
#         i+=1

#     fin = time.clock()
#     t.append(fin-start)

# print("Time: %f from %s" % (sum(t)/len(t), t))


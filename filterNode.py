'''
Filters the provided node by provided parameters
Used in gbrEagleFilterNode
'''
import numpy as np
from eagleFilter import eagleFilter as eagle
import json
import logging as log
from datetime import datetime, timedelta
from itertools import count, groupby

FILTER_MIN_WINDOW = 10 #days

""" Sets values to False if there are more than n consecutive True values
    This allows for us to interpolate only the NAN values in the data where there 
        are fewer than n consecutive NANs, only True values in the mask are interpolated """
# appear to be working but could test better
# this could very likely be more efficient
def mask_nan(mask, n):
    #print(mask)
    i=0
    while i<(len(mask)-n-1):
        if((mask[i:i+n]==True).all()):
            mask[i:i+n]=False
            i+=(n)
            while(mask[i]==True and len(mask) < i):
                mask[i] = False
                i+=1
        else:
            i+=1

    #print(mask) 
    return mask  

"""takes the node, loads data and runs all  filters on it"""
def run_filter(input_data, upper_threhold, lower_threhold, changing_rate,  
                start_time, finish_time, 
                refANode, refBNode, refCNode, refDNode, SQINode):

    # Run the reference filter if ref is defined
    # if(refANode and refBNode and refCNode and refDNode and SQINode):
    #     input_data_ref = reference_filter(input_data, refANode, refBNode, refCNode, refDNode, SQINode, start_time, finish_time)
    # else:
    #     input_data_ref = input_data.copy()
    input_data_ref = input_data.copy()

    # Directly from lambda_function.py
    if(not (np.isnan(upper_threhold)and np.isnan(lower_threhold))):
        input_data_filtered = threshold_filter(input_data_ref, upper_threhold, lower_threhold)
    else:
        input_data_filtered = input_data_ref.copy()
        log.info("No threshold filter")

    input_data_filtered_seocnd = changing_rate_filter(input_data_filtered,changing_rate)

    # may be unneccesary 
    # convert pandas data frame to list
    f = input_data_filtered_seocnd.tolist()
    return f

def threshold_filter(input_data, upper_threhold, lower_threhold):
    # apply threshold filter
    # filtered data is replaced by NAN

    mask = np.ones(len(input_data)) !=  np.ones(len(input_data))
    if(not np.isnan(lower_threhold)):
        mask |= (input_data<lower_threhold)
    if(not np.isnan(upper_threhold)):
        mask |= (input_data>upper_threhold)
    # this only produces a warning but remove for now
    #mask1 = (input_data<lower_threhold)|(input_data>upper_threhold)
    #print("---------", np.all(mask1==mask))

    input_data_filtered = input_data.copy()
    input_data_filtered[mask] = np.NAN
    
    # interpolate
    # linear interpolation to remove NAN
    
    mask = np.isnan(input_data_filtered)
    #print("a",np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), input_data_filtered[~mask]))
    mask = mask_nan(mask,5) # change n to change size of uninterpolated consecutive nan
    #print("b",np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), input_data_filtered[~mask]))
    input_data_filtered[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), input_data_filtered[~mask])
    return input_data_filtered

def changing_rate_filter(input_data_filtered,changing_rate):
    # Changing rate filter - spliced
    # filtered data is replaced by NAN
    # find intervals of non-nan data
    indecies = np.asarray(np.argwhere(~np.isnan(input_data_filtered)))[:,0]
    ranges = [] # will be an array of start and end index for non nan blocks
    for v,g in groupby(indecies, lambda n, c=count(): n-next(c)):
        k = list(g)
        ranges.append([k[0], k[-1]+1])

    input_data_filtered_seocnd = input_data_filtered.copy()
    # calculate the changing rate filter for each range seperately
    for i in range(0, len(ranges)):
        data = input_data_filtered[ranges[i][0]:ranges[i][1]]
        diff_index = np.diff(data)

        mask_diff = np.greater(abs(diff_index), changing_rate, where=~np.isnan(diff_index))
        mask_diff = np.insert(mask_diff, 0, False)

        data_seocnd = data.copy()
        data_seocnd[mask_diff] = np.NAN

        # interpolate
        # linear interpolation to remove NAN
        mask = np.isnan(data_seocnd) #is necessary because some NAN values not from changing rage mask
        mask = mask_nan(mask,5)
        data_seocnd[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data_seocnd[~mask])
        input_data_filtered_seocnd[ranges[i][0]:ranges[i][1]] = data_seocnd    

    return input_data_filtered_seocnd

def reference_filter(input_data, refANode, refBNode, refCNode, refDNode, SQINode, start_time, finish_time):
    ea = eagle() # new instance but could pass around

    # the following takes a long time
    refA = np.asarray(ea.getData(refANode, start_time, finish_time + timedelta(seconds=1)))[:,1].astype(float)
    refB = np.asarray(ea.getData(refBNode, start_time, finish_time + timedelta(seconds=1)))[:,1].astype(float)
    refC = np.asarray(ea.getData(refCNode, start_time, finish_time + timedelta(seconds=1)))[:,1].astype(float)
    refD = np.asarray(ea.getData(refDNode, start_time, finish_time + timedelta(seconds=1)))[:,1].astype(float)
    SQI = np.asarray(ea.getData(SQINode, start_time, finish_time + timedelta(seconds=1)))[:,1].astype(float)

    # mask values of RefD > 13000
    mask  = refD < 13000 
    # mask other reference less than 150
    mask |= ((refA < 150)|(refB < 150)|(refC < 150))
    # SQI < 0.8 or SQI > 1
    mask |= (SQI<0.8)|(SQI>1)

    mask |= (np.abs(refD - refA) > 7000)|(np.abs(refD - refB) > 7000)|(np.abs(refD - refC) > 7000)

    # if(any(refD < 13000)):
    #     print("1")
    # if(any(((refA < 150)|(refB < 150)|(refC < 150)))):
    #      print("2")
    # if(any((SQI<0.8)|(SQI>1))):
    #      print("3")
    # if(any((np.abs(refD - refA) > 7000)|(np.abs(refD - refB) > 7000)|(np.abs(refD - refC) > 7000))):
    #      print("4")

    input_data_filtered = input_data.copy()
    input_data_filtered[mask] = np.NAN
    
    # interpolate
    mask = np.isnan(input_data_filtered)
    mask = mask_nan(mask,5)
    input_data_filtered[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), input_data_filtered[~mask])
    return input_data_filtered

""" gets data from eagle io and if there is new data filters it
and reuloads it 

Could have this in run filter or in the filter loop but not sure where so 
having seperate
"""
def filter_data(source_node, dest_node, refANode, refBNode, refCNode, refDNode, SQINode, upper_threhold, lower_threhold, changing_rate):
    # get input data from node and convert
    ea = eagle()
    global FILTER_MIN_WINDOW

    source_metadata = ea.getLocationMetadata(source_node)
    dest_metadata = ea.getLocationMetadata(dest_node)
    # what do these look like?
    #print(source_metadata)

    # check if flag for empty stream is set
    if(source_metadata['currentTime'] == 0):
        log.warning("No data in source node. No filtering.")
        return 0
   
    if(dest_metadata['currentTime'] == 0):
        # use all of the available data
        log.warning("Empty time fields in destination node, requesting all source data.")
        dest_metadata['currentTime'] = source_metadata['oldestTime']# + timedelta(days=FILTER_MIN_WINDOW)

    #print("Time difference: ", dest_metadata['currentTime'],  source_metadata['currentTime'], dest_metadata['currentTime']< source_metadata['currentTime'])
    # dest_metadata['currentTime'] = source_metadata['currentTime'] - timedelta(minutes=1)
    # check that there is new data
    if dest_metadata['currentTime'] < source_metadata['currentTime']:
        # get all new data
        start_time =  dest_metadata['currentTime'] - timedelta(days=FILTER_MIN_WINDOW)
        finish_time = source_metadata['currentTime']
       
        # the get in the eagle api is not inclusive so add one second to finish_time 
        # #     so that all the data including the last point is retrieved and the process will not be repeated 
        data = ea.getData(source_node, start_time, finish_time + timedelta(seconds=1)) # add one min here
        print("Filtering: ", start_time, finish_time, len(data), "; time_dif: ", start_time-finish_time, data[0])

        # format data
        input_data = np.asarray(data)[:,1]   
        
        input_data = input_data.astype(float)

        #all new data is nan so no filtering occurs
        if(np.isnan(input_data).all()):
            return 0
        
        # run all realtime filters
        filtered_data = run_filter(input_data, upper_threhold, 
                lower_threhold, changing_rate, start_time, finish_time, 
                refANode, refBNode, refCNode, refDNode, SQINode)

        # Create JTS JSON time series of filtered data  
        ts = ea.createTimeSeriesJSON(data,filtered_data)

        # update destination on Eagle with filtered data
        res = ea.updateData(dest_node, ts)

        return 1
    print("No new data, no filtering occurred", dest_metadata['currentTime'], source_metadata['currentTime'] )
    return 0

def main(event, context):
    log.basicConfig(level=log.WARNING)

    ''' only required for sns '''
    
    payload = event['Records'][0]['Sns']['Message']
    event = json.loads(payload)
    
    ''' '''

    source_node = event['source_node']
    dest_node = event['dest_node']
    upper_threshold = float(event['upper_threshold'])
    lower_threshold = float(event['lower_threshold'])
    changing_rate = float(event['changing_rate'])
    try:
        refANode = event['refANode']
        refBNode = event['refBNode']
        refCNode = event['refCNode']
        refDNode = event['refDNode']
        SQINode = event['SQINode']
    except:
        refANode = None
        refBNode = None
        refCNode = None
        refDNode = None
        SQINode = None
    
    print("Processing ", source_node, dest_node)
    res = filter_data(source_node, dest_node, 
            refANode, refBNode, refCNode, refDNode, SQINode,
            upper_threshold, lower_threshold, changing_rate)
    if(res == 1):
        response = {
                "statusCode": 200,
                "body": ''
            }
    
        return response


def run():
    testEvent = {'Records': [{'EventSource': 'aws:sns', 
            'EventVersion': '1.0', 'EventSubscriptionArn': 'arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate:1cc5186a-04cc-430a-8065-fa438521d082', 'Sns': {'Type': 'Notification', 'MessageId': 'bc85683f-2efc-50c6-8314-3d51aff722d2', 'TopicArn': 'arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate', 
            'Subject': None, 
            'Message': '{"source_node": "5c3578fc1bbcf10f7880ca5f", "dest_node": "5ca2a9604c52c40f17064db0", "refANode": "5c3578fc1bbcf10f7880ca62", "refBNode": "5c3578fc1bbcf10f7880ca63", "refCNode": "5c3578fc1bbcf10f7880ca64", "refDNode": "5c3578fc1bbcf10f7880ca65", "SQINode": "5c3578fc1bbcf10f7880ca61", "upper_threshold": "1", "lower_threshold": "0.05", "changing_rate": "0.05"}', 
            'Timestamp': '2019-06-03T01:58:35.515Z', 'SignatureVersion': '1', 'Signature': 'MD2dPjKLTGTijU1s+vPuE699sSM7vquQHQFpVBtqECLEX+4psmZeT7oAMSZY5yCAtS2QKesiE4/lR9ezBENfmmTy/TrWyqguyY+4RO121nzlMWN3FN/IPdbNJU2yvsYby7//PwIJDvgN2KgoAhZPoW92bJtFAxOlMKmnNSsfCPM7lH0FF4M2pyvmzbyauFoFhJfdr0hRWfcPnmmMSusr8rc9Y0wdEtR37qexQ99GR8w2KWMZE8VWPNc8ZdXSeE3sLv7floxaxCIqWcS3nm6pJiN/B0YzDBIJvVEIa492qKm8lPd34MCRG6lLH05VJw3KwkOQLbabpJoP43lKhDZdkQ==', 'SigningCertUrl': 'https://sns.ap-southeast-2.amazonaws.com/SimpleNotificationService-6aad65c2f9911b05cd53efda11f913f9.pem', 'UnsubscribeUrl': 'https://sns.ap-southeast-2.amazonaws.com/?Action=Unsubscribe&SubscriptionArn=arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate:1cc5186a-04cc-430a-8065-fa438521d082', 'MessageAttributes': {}}}]}
                
    main(testEvent, None)
    f = open("output.txt", 'a')
    f.write("Ran")


if __name__ == "__main__":
    run()
    # # testEvent = {
    # #             "source_node": "5b177a5de4b05e726c7eeecc",
    # #             "dest_node": "5ca2a9604c52c40f17064dafa",
    # #             "upper_threshold": 2,
    # #             "lower_threshold": 1,
    # #             "changing_rate": 0.5
    # #             }
    # # output on column 3
    # testEvent = {'Records': [{'EventSource': 'aws:sns', 
    #             'EventVersion': '1.0', 'EventSubscriptionArn': 'arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate:1cc5186a-04cc-430a-8065-fa438521d082', 'Sns': {'Type': 'Notification', 'MessageId': 'bc85683f-2efc-50c6-8314-3d51aff722d2', 'TopicArn': 'arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate', 
    #             'Subject': None, 
    #             'Message': '{"source_node": "5c3578fc1bbcf10f7880ca5f", "dest_node": "5ca2a9604c52c40f17064db0", "refANode": "5c3578fc1bbcf10f7880ca62", "refBNode": "5c3578fc1bbcf10f7880ca63", "refCNode": "5c3578fc1bbcf10f7880ca64", "refDNode": "5c3578fc1bbcf10f7880ca65", "SQINode": "5c3578fc1bbcf10f7880ca61", "upper_threshold": "1", "lower_threshold": "0.05", "changing_rate": "0.05"}', 
    #             'Timestamp': '2019-06-03T01:58:35.515Z', 'SignatureVersion': '1', 'Signature': 'MD2dPjKLTGTijU1s+vPuE699sSM7vquQHQFpVBtqECLEX+4psmZeT7oAMSZY5yCAtS2QKesiE4/lR9ezBENfmmTy/TrWyqguyY+4RO121nzlMWN3FN/IPdbNJU2yvsYby7//PwIJDvgN2KgoAhZPoW92bJtFAxOlMKmnNSsfCPM7lH0FF4M2pyvmzbyauFoFhJfdr0hRWfcPnmmMSusr8rc9Y0wdEtR37qexQ99GR8w2KWMZE8VWPNc8ZdXSeE3sLv7floxaxCIqWcS3nm6pJiN/B0YzDBIJvVEIa492qKm8lPd34MCRG6lLH05VJw3KwkOQLbabpJoP43lKhDZdkQ==', 'SigningCertUrl': 'https://sns.ap-southeast-2.amazonaws.com/SimpleNotificationService-6aad65c2f9911b05cd53efda11f913f9.pem', 'UnsubscribeUrl': 'https://sns.ap-southeast-2.amazonaws.com/?Action=Unsubscribe&SubscriptionArn=arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate:1cc5186a-04cc-430a-8065-fa438521d082', 'MessageAttributes': {}}}]}
    
    # # testEvent = {'Records': [{'EventSource': 'aws:sns', 
    # #             'EventVersion': '1.0', 'EventSubscriptionArn': 'arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate:1cc5186a-04cc-430a-8065-fa438521d082', 'Sns': {'Type': 'Notification', 'MessageId': 'bc85683f-2efc-50c6-8314-3d51aff722d2', 'TopicArn': 'arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate', 
    # #             'Subject': None, 
    # #             'Message': '{"source_node": "5c3578fc1bbcf10f7880ca5f", "dest_node": "5ca2a9604c52c40f17064db0", "upper_threshold": "1", "lower_threshold": "0.05", "changing_rate": "0.05"}', 
    # #             'Timestamp': '2019-06-03T01:58:35.515Z', 'SignatureVersion': '1', 'Signature': 'MD2dPjKLTGTijU1s+vPuE699sSM7vquQHQFpVBtqECLEX+4psmZeT7oAMSZY5yCAtS2QKesiE4/lR9ezBENfmmTy/TrWyqguyY+4RO121nzlMWN3FN/IPdbNJU2yvsYby7//PwIJDvgN2KgoAhZPoW92bJtFAxOlMKmnNSsfCPM7lH0FF4M2pyvmzbyauFoFhJfdr0hRWfcPnmmMSusr8rc9Y0wdEtR37qexQ99GR8w2KWMZE8VWPNc8ZdXSeE3sLv7floxaxCIqWcS3nm6pJiN/B0YzDBIJvVEIa492qKm8lPd34MCRG6lLH05VJw3KwkOQLbabpJoP43lKhDZdkQ==', 'SigningCertUrl': 'https://sns.ap-southeast-2.amazonaws.com/SimpleNotificationService-6aad65c2f9911b05cd53efda11f913f9.pem', 'UnsubscribeUrl': 'https://sns.ap-southeast-2.amazonaws.com/?Action=Unsubscribe&SubscriptionArn=arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate:1cc5186a-04cc-430a-8065-fa438521d082', 'MessageAttributes': {}}}]}
    # # main(testEvent, None)
    # import time
    # start = time.clock()
    # main(testEvent, None)
    # fin = time.clock()
    # print("Time: %f" % (fin-start))

    # # powershell -Command Measure-Command {python filterNode.py}
    # # python -m timeit -n 1 -s "from filterNode import run" "run()"




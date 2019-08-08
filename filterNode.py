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

"""
Note:
    Currently there may be a timeout if there are too many new values to process
    due to the time it takes to get data from eagle for all reference nodes 
    (about an extra 1 second for 10 days of data)

    If the filter is not able to run for some time or if more than some value of 
    data comes through in the interval between filtering it may still time out
    This is for now unaddressed
"""


""" Sets values to False if there are more than n consecutive True values
    This allows for us to interpolate only the NAN values in the data where there 
        are fewer than n consecutive NANs, only True values in the mask are interpolated """
# this could likely be more efficient
def mask_nan(mask, n):
    i=0
    while i<(len(mask)-n-1):
        if((mask[i:i+n]==True).all()):
            mask[i:i+n]=False
            i+=(n)
            while(mask[i]==True and i<len(mask) ):
                mask[i] = False
                i+=1
        else:
            i+=1

    return mask  

"""
also returns information relevant for ml
"""
def mask_nan_info(mask, n):
    print(mask)
    gaps = []
    i=0
    while i<(len(mask)-n-1):
        if((mask[i:i+n]==True).all()):
            # mask[i:i+n]=False
            gap_info = [i] # first missing
            i+=(n)
            while(mask[i]==True and i<len(mask)):
                # mask[i] = False
                i+=1
            gap_info.append(i) # end of gap
            print("gi", gap_info)
            gaps.append(gap_info)
        else:
            i+=1

    return mask, gaps     

""" Calculate the mask for filtering values from regerence values, 
    True values in the mask are the data filtered out"""
def find_reference_mask(input_data, input_dates, refA, refB, refC, refD, SQI):
    refD = np.array(refD) #k
    refD_date = np.array(refD[:,0]).astype('datetime64')
    refD = refD[:,1].astype(float)

    SQI = np.array(SQI) #j
    SQI_date = np.array(SQI[:,0]).astype('datetime64')
    SQI = SQI[:,1].astype(float)

    refA = np.array(refA) #m
    refA_date = np.array(refA[:,0]).astype('datetime64')
    refA = refA[:,1].astype(float)

    refB = np.array(refB) #n
    refB_date = np.array(refB[:,0]).astype('datetime64')
    refB = refB[:,1].astype(float)

    refC = np.array(refC) #p
    refC_date = np.array(refC[:,0]).astype('datetime64')
    refC = refC[:,1].astype(float)

    input_dates = np.array(input_dates).astype('datetime64')
    mask = np.full(len(input_dates), False)


    i = j = k = m = n = p = 0
    while(i<len(input_dates)):
        kn = 0
        if(input_dates[i] == SQI_date[j]):
            mask[i] =  (SQI[j]<0.8)|(SQI[j]>1)
        elif(input_dates[i] < SQI_date[j]):
            mask[i] = False
            j -= 1
        elif(input_dates[i] > SQI_date[j]): 
            j+=1

        if(input_dates[i] == refD_date[k]):
            mask[i] |=  refD[k]<13000
        elif(input_dates[i] < refD_date[k]):
            mask[i] |= False
            kn += 1
        elif(input_dates[i] > refD_date[k]): 
            kn-=1

        if(input_dates[i] == refA_date[m]):
            mask[i] |=  (refA[m]<150)|(np.abs(refD[m] - refA[m]) > 7000)
        elif(input_dates[i] < refA_date[m]):
            mask[i] |= False
            m -= 1

        if(input_dates[i] == refB_date[n]):
            mask[i] |=  (refB[n]<150)|(np.abs(refD[n] - refB[n]) > 7000)
        elif(input_dates[i] < refB_date[n]):
            mask[i] |= False
            n -= 1

        if(input_dates[i] == refC_date[p]):
            mask[i] |=  (refC[p]<150)|(np.abs(refD[p] - refC[p]) > 7000)
        elif(input_dates[i] < refC_date[p]):
            mask[i] |= False
            p -= 1

        i+=1
        j+=1
        k=k+1-kn
        m+=1
        n+=1
        p+=1

    return mask
        

"""Given node, loads data and runs all filters on it"""
def run_filter(input_data, upper_threshold, lower_threshold, changing_rate,  
                start_time, finish_time, 
                refANode, refBNode, refCNode, refDNode, SQINode, input_dates):

    # Run the reference filter if ref is defined
    if(refANode and refBNode and refCNode and refDNode and SQINode):
        input_data_ref = reference_filter(input_data, refANode, refBNode, refCNode, 
                        refDNode, SQINode, start_time, finish_time, input_dates)
    else:
        input_data_ref = input_data.copy()

    # Directly from lambda_function.py
    if(not (np.isnan(upper_threshold) and np.isnan(lower_threshold))):
        input_data_filtered = threshold_filter(input_data_ref, upper_threshold, lower_threshold)
    else:
        # filter is not run if neither threshold is set
        input_data_filtered = input_data_ref.copy()
        log.info("No threshold filter")

    input_data_filtered_seocnd= changing_rate_filter(input_data_filtered,changing_rate)

    mask = np.isnan(input_data_filtered_seocnd)
    mask, gaps = mask_nan_info(mask, 5)
    print("chekcing gaps", gaps)
    pre_length = 24
    info = {}
    for gap in gaps:
        gap_length = gap[1]-gap[0]
        if(gap[0]>pre_length):
            pre_data = input_data_filtered_seocnd[gap[0]-pre_length:gap[0]]
            pre_index = input_dates[gap[0]-pre_length:gap[0]]
        else: 
            pre_data = input_data_filtered_seocnd[0:gap[0]]
            pre_index = input_dates[0:gap[0]]
        gap_index = input_dates[gap[0]:gap[1]]

        info[gap[0]]= {'length': gap_length, "input": pre_data.tolist(), 
                        "input_dates": pre_index.tolist(),  "pred_dates": gap_index.tolist()}
    # print(info)
        
    # convert to json
    gap_info_file = open("gap_info.json", "w")
    json.dump(info, gap_info_file)
    # gap_info = json.dumps(info)
    # upload to s3
    #ea.uploadDataAWSJSON()

   
    # may be unneccesary 
    # convert pandas data frame to list
    f = input_data_filtered_seocnd.tolist()
    return f

def threshold_filter(input_data, upper_threshold, lower_threshold):
    # apply threshold filter
    # filtered data is replaced by NAN

    mask = np.ones(len(input_data)) !=  np.ones(len(input_data))
    if(not np.isnan(lower_threshold)):
        mask |= np.less(input_data, lower_threshold, where=~np.isnan(input_data))
    if(not np.isnan(upper_threshold)):
        mask|=np.greater(input_data, upper_threshold, where=~np.isnan(input_data))

    input_data_filtered = input_data.copy()
    input_data_filtered[mask] = np.NAN
    
    # interpolate
    # linear interpolation to remove NAN
    mask = np.isnan(input_data_filtered)
    mask = mask_nan(mask,5) # change n to change size of uninterpolated consecutive nan
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

def reference_filter(input_data, refANode, refBNode, refCNode, refDNode, 
                    SQINode, start_time, finish_time, input_dates):
    ea = eagle() # new instance but could pass around

    refA= ea.getData(refANode, start_time, finish_time + timedelta(seconds=1))
    refB = ea.getData(refBNode, start_time, finish_time + timedelta(seconds=1))
    refC = ea.getData(refCNode, start_time, finish_time + timedelta(seconds=1))
    refD = ea.getData(refDNode, start_time, finish_time + timedelta(seconds=1))
    SQI = ea.getData(SQINode, start_time, finish_time + timedelta(seconds=1))
   
    mask = find_reference_mask(input_data, input_dates, refA, refB, refC, refD, SQI)


    if (len(mask) != len(input_data)):
        log.warning("The reference and value streams have different number of data points \n No reference filter applied")
        return input_data
    # filter from mask
    input_data_filtered = input_data.copy()
    input_data_filtered[mask] = np.NAN
    
    # interpolate
    mask = np.isnan(input_data_filtered)
    mask = mask_nan(mask,5)
    input_data_filtered[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), input_data_filtered[~mask])
    return input_data_filtered

""" gets data from eagle io and if there is new data filters it
and reuploads it 
"""
def filter_data(source_node, dest_node, refANode, refBNode, refCNode, refDNode, 
                SQINode, upper_threshold, lower_threshold, changing_rate):
    # get input data from node and convert
    ea = eagle()
    global FILTER_MIN_WINDOW

    source_metadata = ea.getLocationMetadata(source_node)
    dest_metadata = ea.getLocationMetadata(dest_node)

    # check if flag for empty stream is set
    if(source_metadata['currentTime'] == 0):
        log.warning("No data in source node. No filtering.")
        return 0

    # destination node is empty
    if(dest_metadata['currentTime'] == 0):
        # use all of the available data
        log.warning("Empty time fields in destination node, requesting all source data.")
        dest_metadata['currentTime'] = source_metadata['oldestTime']

    ### for testing filtering all
    # dest_metadata['currentTime'] = source_metadata['oldestTime']
    ### for testing when already filtered
    dest_metadata['currentTime'] = source_metadata['currentTime'] - timedelta(hours=3)
  

    # check that there is new data
    if dest_metadata['currentTime'] < source_metadata['currentTime']:
        # get all new data plus a delay window
        start_time =  dest_metadata['currentTime'] #- timedelta(days=FILTER_MIN_WINDOW)
        finish_time = source_metadata['currentTime']
        print(start_time, finish_time)
       
        # the get in the eagle api is not inclusive so add one second to finish_time 
        # #     so that all the data including the last point is retrieved and the process will not be repeated 
        data = ea.getData(source_node, start_time, finish_time + timedelta(seconds=1)) 
        print("Filtering: ", start_time, finish_time, "Length: ", len(data), "Time_dif: ", start_time-finish_time)
        # format data
        input_data = np.asarray(data)[:,1]   
        
        input_data = input_data.astype(float)
        input_dates = np.asarray(data)[:,0]  
        input_data = np.array([1, 1, float('nan'),  float('nan'),  float('nan'),  float('nan'),  float('nan'),  float('nan'),  float('nan'), 1])
        
        start_time = datetime.strptime(input_dates[0], '%Y-%m-%dT%H:%M:%S')
        finish_time = datetime.strptime(input_dates[-1], '%Y-%m-%dT%H:%M:%S')
        
        # all new data is nan so no filtering occurs
        if(np.isnan(input_data).all()):
            return 0
        
        # run all realtime filters
        filtered_data = run_filter(input_data, upper_threshold, 
                lower_threshold, changing_rate, start_time, finish_time, 
                refANode, refBNode, refCNode, refDNode, SQINode, input_dates)

        # Create JTS JSON time series of filtered data  
        ts = ea.createTimeSeriesJSON(data,filtered_data)

        # update destination on Eagle with filtered data
        # res = ea.updateData(dest_node, ts)

        return 1
    log.warning("No new data, no filtering occurred")
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

######

def run():
    testEvent = {'Records': [{'EventSource': 'aws:sns', 
            'EventVersion': '1.0', 'EventSubscriptionArn': 'arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate:1cc5186a-04cc-430a-8065-fa438521d082', 'Sns': {'Type': 'Notification', 'MessageId': 'bc85683f-2efc-50c6-8314-3d51aff722d2', 'TopicArn': 'arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate', 
            'Subject': None, 
            'Message': '{"interval": "60", source_node": "5c3578fc1bbcf10f7880ca5f", "dest_node": "5ca2a9604c52c40f17064db0", "refANode": "5c3578fc1bbcf10f7880ca62", "refBNode": "5c3578fc1bbcf10f7880ca63", "refCNode": "5c3578fc1bbcf10f7880ca64", "refDNode": "5c3578fc1bbcf10f7880ca65", "SQINode": "5c3578fc1bbcf10f7880ca61", "upper_threshold": "1", "lower_threshold": "0", "changing_rate": "0.05"}', 
            'Timestamp': '2019-06-03T01:58:35.515Z', 'SignatureVersion': '1', 'Signature': 'MD2dPjKLTGTijU1s+vPuE699sSM7vquQHQFpVBtqECLEX+4psmZeT7oAMSZY5yCAtS2QKesiE4/lR9ezBENfmmTy/TrWyqguyY+4RO121nzlMWN3FN/IPdbNJU2yvsYby7//PwIJDvgN2KgoAhZPoW92bJtFAxOlMKmnNSsfCPM7lH0FF4M2pyvmzbyauFoFhJfdr0hRWfcPnmmMSusr8rc9Y0wdEtR37qexQ99GR8w2KWMZE8VWPNc8ZdXSeE3sLv7floxaxCIqWcS3nm6pJiN/B0YzDBIJvVEIa492qKm8lPd34MCRG6lLH05VJw3KwkOQLbabpJoP43lKhDZdkQ==', 'SigningCertUrl': 'https://sns.ap-southeast-2.amazonaws.com/SimpleNotificationService-6aad65c2f9911b05cd53efda11f913f9.pem', 'UnsubscribeUrl': 'https://sns.ap-southeast-2.amazonaws.com/?Action=Unsubscribe&SubscriptionArn=arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate:1cc5186a-04cc-430a-8065-fa438521d082', 'MessageAttributes': {}}}]}
                
    main(testEvent, None)
    f = open("output.txt", 'a')
    f.write("Ran")

def test_resample():
    input_data = []
    input_dates = ['2019-05-01 00:00:00', '2019-05-01 01:00:00', '2019-05-01 02:00:00', '2019-05-01 03:00:00']
    data =  [['2019-05-01 00:00:00',1], ['2019-05-01 02:00:00', 3], ['2019-05-01 03:00:00', 4]]
    fdata = resample(input_data, input_dates, data)
    print(fdata)


if __name__ == "__main__":

    # output on column 3
    testEventRef = {'Records': [{'EventSource': 'aws:sns', 
                'EventVersion': '1.0', 'EventSubscriptionArn': 'arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate:1cc5186a-04cc-430a-8065-fa438521d082', 'Sns': {'Type': 'Notification', 'MessageId': 'bc85683f-2efc-50c6-8314-3d51aff722d2', 'TopicArn': 'arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate', 
                'Subject': None, 
                'Message': '{"source_node": "5c3578fc1bbcf10f7880ca5f", "dest_node": "5ca2a9604c52c40f17064db0", "refANode": "5c3578fc1bbcf10f7880ca62", "refBNode": "5c3578fc1bbcf10f7880ca63", "refCNode": "5c3578fc1bbcf10f7880ca64", "refDNode": "5c3578fc1bbcf10f7880ca65", "SQINode": "5c3578fc1bbcf10f7880ca61", "upper_threshold": "2", "lower_threshold": "0", "changing_rate": "0.5"}', 
                'Timestamp': '2019-06-03T01:58:35.515Z', 'SignatureVersion': '1', 'Signature': 'MD2dPjKLTGTijU1s+vPuE699sSM7vquQHQFpVBtqECLEX+4psmZeT7oAMSZY5yCAtS2QKesiE4/lR9ezBENfmmTy/TrWyqguyY+4RO121nzlMWN3FN/IPdbNJU2yvsYby7//PwIJDvgN2KgoAhZPoW92bJtFAxOlMKmnNSsfCPM7lH0FF4M2pyvmzbyauFoFhJfdr0hRWfcPnmmMSusr8rc9Y0wdEtR37qexQ99GR8w2KWMZE8VWPNc8ZdXSeE3sLv7floxaxCIqWcS3nm6pJiN/B0YzDBIJvVEIa492qKm8lPd34MCRG6lLH05VJw3KwkOQLbabpJoP43lKhDZdkQ==', 'SigningCertUrl': 'https://sns.ap-southeast-2.amazonaws.com/SimpleNotificationService-6aad65c2f9911b05cd53efda11f913f9.pem', 'UnsubscribeUrl': 'https://sns.ap-southeast-2.amazonaws.com/?Action=Unsubscribe&SubscriptionArn=arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate:1cc5186a-04cc-430a-8065-fa438521d082', 'MessageAttributes': {}}}]}
    
    testEvent = {'Records': [{'EventSource': 'aws:sns', 
                'EventVersion': '1.0', 'EventSubscriptionArn': 'arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate:1cc5186a-04cc-430a-8065-fa438521d082', 'Sns': {'Type': 'Notification', 'MessageId': 'bc85683f-2efc-50c6-8314-3d51aff722d2', 'TopicArn': 'arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate', 
                'Subject': None, 
                'Message': '{"source_node": "5c3578fc1bbcf10f7880ca5f", "dest_node": "5ca2a9604c52c40f17064db0", "upper_threshold": "1", "lower_threshold": "0.05", "changing_rate": "0.05"}', 
                'Timestamp': '2019-06-03T01:58:35.515Z', 'SignatureVersion': '1', 'Signature': 'MD2dPjKLTGTijU1s+vPuE699sSM7vquQHQFpVBtqECLEX+4psmZeT7oAMSZY5yCAtS2QKesiE4/lR9ezBENfmmTy/TrWyqguyY+4RO121nzlMWN3FN/IPdbNJU2yvsYby7//PwIJDvgN2KgoAhZPoW92bJtFAxOlMKmnNSsfCPM7lH0FF4M2pyvmzbyauFoFhJfdr0hRWfcPnmmMSusr8rc9Y0wdEtR37qexQ99GR8w2KWMZE8VWPNc8ZdXSeE3sLv7floxaxCIqWcS3nm6pJiN/B0YzDBIJvVEIa492qKm8lPd34MCRG6lLH05VJw3KwkOQLbabpJoP43lKhDZdkQ==', 'SigningCertUrl': 'https://sns.ap-southeast-2.amazonaws.com/SimpleNotificationService-6aad65c2f9911b05cd53efda11f913f9.pem', 'UnsubscribeUrl': 'https://sns.ap-southeast-2.amazonaws.com/?Action=Unsubscribe&SubscriptionArn=arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate:1cc5186a-04cc-430a-8065-fa438521d082', 'MessageAttributes': {}}}]}

    import time
    start = time.clock()
    main(testEventRef, None)
    fin = time.clock()
    print("Time: %f sec" % (fin-start))

    # powershell -Command Measure-Command {python filterNode.py}
    # python -m timeit -n 1 -s "from filterNode import run" "run()"



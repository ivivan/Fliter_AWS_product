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

FILTER_MIN_WINDOW = 10  # days
MAX_GAP = 5

"""
Note:
    Currently there may be a timeout if there are too many new values to process
    due to the time it takes to get data from eagle for all reference nodes 
    (about an extra 1 second for 10 days of data)

    If the filter is not able to run for some time or if more than some value of 
    data comes through in the interval between filtering it may still time out
    This is for now unaddressed
"""

# this could likely be more efficient
def mask_nan(mask, n):
    """ Sets values to False if there are more than n consecutive True values

    This allows for us to interpolate only the NAN values in the data where
    there are fewer than n consecutive NANs, only True values in the mask
    are interpolated 
    """
    i=0
    while i < (len(mask)-n-1):
        # if all of the next n values are true set them to false and move
        # forward by n
        if((mask[i:i+n] == True).all()):
            mask[i:i+n] = False
            i += (n)

            # check for any further consecutive true values
            while(i < len(mask) and mask[i] == True):
                mask[i] = False
                i += 1
        else:
            i += 1
    return mask  


def find_reference_mask_nico(input_data, input_dates, refA, refB, refC, refD, SQI):
    """ Calculate the mask for filtering values from reference values for nico
    sensors.

    True values in the mask are the data filtered out
    """
    # seperate the dates and values into two lists for each reference 
    refD = np.array(refD) #k loop variable
    refD_date = np.array(refD[:,0]).astype('datetime64')
    refD = refD[:,1].astype(float) # convert from string to float

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

    # Not every data point has a corrosponding reference value so dates must
    # be checked before comparing.
    # If a value in any of the streams is nan it is not present in the data at 
    # all. So we need to make sure the date for the data and the reference value 
    # match before masking the value. To be able to move around values in each 
    # array, each reference node has its own loop variable.
    i = j = k = m = n = p = 0

    while(i < len(input_dates)):
        # reference value D is used in multiple places so even though it may not 
        # match dates in one case it may in other conditions. This means it 
        # cannot just be incremented when the dates first don't match and has to
        # be incremented at the end. kn is how much the loop variable for 
        # reference D (k) needs to be incremented.
        kn = 0

        if (input_dates[i] == SQI_date[j]):
            # if the dates match apply the appropriate test
            mask[i] =  (SQI[j]<0.5)|(SQI[j]>1)
        elif (input_dates[i] < SQI_date[j]):
            # if the input date is less than the reference date there is no
            # reference value for this date, it cannot be filtered out. We also
            # need to re-check this reference with the next input date 
            mask[i] = False
            j -= 1
        elif (input_dates[i] > SQI_date[j]): 
            # if the input date is greater than the reference we check the next 
            # reference
            j+=1

        if (input_dates[i] == refD_date[k]):
            mask[i] |= refD[k] < 13000
        elif (input_dates[i] < refD_date[k]):
            mask[i] |= False
            kn += 1 # D is a special case
        elif (input_dates[i] > refD_date[k]): 
            kn -= 1

        if (input_dates[i] == refA_date[m]):
            mask[i] |=  (refA[m]<150)|(np.abs(refD[m] - refA[m]) > 7000)
        elif (input_dates[i] < refA_date[m]):
            mask[i] |= False
            m -= 1
        
        if (input_dates[i] == refB_date[n]):
            mask[i] |=  (refB[n]<150)|(np.abs(refD[n] - refB[n]) > 7000)
        elif (input_dates[i] < refB_date[n]):
            mask[i] |= False
            n -= 1

        if (input_dates[i] == refC_date[p]):
            mask[i] |=  (refC[p]<150)|(np.abs(refD[p] - refC[p]) > 7000)
        elif (input_dates[i] < refC_date[p]):
            mask[i] |= False
            p -= 1

        i += 1
        j += 1
        k = k + 1 - kn
        m += 1
        n += 1
        p += 1
 
    return mask

def find_reference_mask_opus(input_data, input_dates, abs360, abs210, SQI):
    """ Calculate the mask for filtering values from reference values for opus
    sensors.

    True values in the mask are the data filtered out
    """
    # seperate the dates and values into two lists for each reference 
    SQI = np.array(SQI) #j loop variable
    SQI_date = np.array(SQI[:,0]).astype('datetime64')
    SQI = SQI[:,1].astype(float)

    abs360 = np.array(abs360) #m
    abs360_date = np.array(abs360[:,0]).astype('datetime64')
    abs360 = abs360[:,1].astype(float)

    abs210 = np.array(abs210) #n
    abs210_date = np.array(abs210[:,0]).astype('datetime64')
    abs210 = abs210[:,1].astype(float)

    input_dates = np.array(input_dates).astype('datetime64')
    mask = np.full(len(input_dates), False)

    # not every data point has a corrosponding reference value so dates must
    # be checked before comparing
    i = j = m = n  = 0

    while (i < len(input_dates)):
        if (input_dates[i] == SQI_date[j]):
            # if the dates match apply the appropriate test
            mask[i] =  (SQI[j]<0.5)|(SQI[j]>1)
        elif (input_dates[i] < SQI_date[j]):
            # if the input date is less than the reference date there is no
            # reference value for this date, it cannot be filtered out. We also
            # need to re-check this reference with the next input date 
            mask[i] = False
            j -= 1
        elif (input_dates[i] > SQI_date[j]):
            # if the input date is greater than the reference we check the next 
            # reference 
            j += 1

        if (input_dates[i] == abs360_date[m]):
            mask[i] |=  (abs360[m]>=0.8)
        elif (input_dates[i] < abs360_date[m]):
            mask[i] |= False
            m -= 1
        
        if (input_dates[i] == abs210_date[n]):
            mask[i] |=  (abs210[n]>=3)
        elif (input_dates[i] < abs210_date[n]):
            mask[i] |= False
            n -= 1

        i += 1
        j += 1
        m += 1
        n += 1
    
    return mask

def run_filter(input_data, upper_threshold, lower_threshold, changing_rate,  
                start_time, finish_time, refANode, refBNode, refCNode, refDNode, 
                SQINode, input_dates, src_read_api_key):
    """ Runs all filters on input data """

    # quality code 0 to start with
    quality = np.zeros(len(input_data))

    # Run the reference filter if nodes are defined
    if(refANode and refBNode and SQINode):
        reference_filtered_data, q1 = reference_filter(input_data, refANode, 
                        refBNode, refCNode, refDNode, SQINode, start_time, 
                        finish_time, input_dates, src_read_api_key)
        quality += q1
    else:
        # the next function still expects reference_filtered_data to be defined
        reference_filtered_data = input_data.copy()


    # Directly from lambda_function.py
    if (not (np.isnan(upper_threshold) and np.isnan(lower_threshold))):
        threshold_filtered_data, q2 = threshold_filter(reference_filtered_data, 
                                    upper_threshold, lower_threshold)
        quality += q2
    else:
        # Filter is not run if neither threshold is set
        threshold_filtered_data = reference_filtered_data.copy()
        log.info("No threshold filter was run")
    

    rate_filtered_data, q3 = changing_rate_filter(threshold_filtered_data,
                                        changing_rate)
    quality += q3 
   
    return rate_filtered_data, quality

def threshold_filter(input_data, upper_threshold, lower_threshold):
    """ Filter values above or below critical values 
    Filtered data is replaced by NAN
    """
    
    # Create an array of false values the same length as input_data
    mask = np.ones(len(input_data)) !=  np.ones(len(input_data))
    
    if (not np.isnan(lower_threshold)):
        # compare only where there is a value
        mask |= np.less_equal(input_data, lower_threshold, where=~np.isnan(input_data))
    if (not np.isnan(upper_threshold)):
        mask|=np.greater_equal(input_data, upper_threshold, where=~np.isnan(input_data))

    # apply quality code with mask of what is filtered by threshold
    quality = np.zeros(len(input_data))
    quality[mask] = 2

    # filter data
    input_data_filtered = input_data.copy()
    input_data_filtered[mask] = np.NAN

    # There may be values that are already NAN or were filtered but not 
    # interpolated by another function so to include these in the n sized gap
    # first re-write the mask as all the values that are nan
    mask = np.isnan(input_data_filtered)
    # gaps of greater than n = 5 are not interpolated, so they are taken out
    # of the mask

    global MAX_GAP
    mask = mask_nan(mask,MAX_GAP) # change n to change size of uninterpolated consecutive nan
    
    if (np.all(mask)):
        log.warning("All data points filtered out by the threshold filter")
    # if(np.flatnonzero(~mask).size==0):
    #     log.warning("All data points filtered out by the threshold filter")
    else:
        # linear interpolation to remove NAN
        input_data_filtered[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), input_data_filtered[~mask])

    return input_data_filtered, quality

def changing_rate_filter(input_data_filtered,changing_rate):
    """ Filter data points with large rates of change
    Filtered data is replaced by NAN

    Since there may be gaps in the data and changing rate does not make sense
    across gaps, this is done in sections spliced by gaps.
    """

    # find intervals of non-nan data
    indecies = np.asarray(np.argwhere(~np.isnan(input_data_filtered)))[:,0]
    ranges = [] # will be an array of start and end index for non nan blocks
    # when the difference between indecies is different break into a new group
    for v,g in groupby(indecies, lambda n, c=count(): n-next(c)):
        k = list(g)
        # save the start and end index for the interval
        ranges.append([k[0], k[-1]+1])

    input_data_filtered_seocnd = input_data_filtered.copy()
    # calculate the changing rate filter for each range seperately

    # mask for all of the data
    total_mask = np.zeros(len(input_data_filtered))
    for i in range(0, len(ranges)):
        data = input_data_filtered[ranges[i][0]:ranges[i][1]]

        # differentiate data to find rate of change
        diff_index = np.diff(data)

        mask_diff = np.greater(abs(diff_index), changing_rate, where=~np.isnan(diff_index))
        # this is required for the mask to be inrterpreted as boolean when 
        # it's applied 
        mask_diff = np.insert(mask_diff, 0, False)

        data_seocnd = data.copy()
        data_seocnd[mask_diff] = np.NAN

        # interpolate
        # linear interpolation to remove NAN
        mask = np.isnan(data_seocnd) #is necessary because some NAN values not from changing rage mask
        global MAX_GAP
        mask = mask_nan(mask,MAX_GAP)
        # If there are still data points to interpolate based on
        if(not np.all(mask)):
            data_seocnd[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data_seocnd[~mask])

        input_data_filtered_seocnd[ranges[i][0]:ranges[i][1]] = data_seocnd    
        total_mask[ranges[i][0]:ranges[i][1]] = mask_diff    

    quality = np.zeros(len(input_data_filtered_seocnd)) 
    quality[total_mask.astype(bool)] = 4

    return input_data_filtered_seocnd, quality

def reference_filter(input_data, refANode, refBNode, refCNode, refDNode, 
                    SQINode, start_time, finish_time, input_dates, src_read_api_key):
    """ Filter data points based on reference values of sensor
    
    Data will be loaded from nodes. If refCNode and refDNode are none, the 
    sensor is an opus sensor and the nodes are interpreted as:
        refANode: abs360Node
        refBNode: abs210Node
    """
    ea = eagle(src_read_api_key) # new instance but could pass around
    
    refA = ea.getData(refANode, start_time, finish_time + timedelta(seconds=1))
    refB = ea.getData(refBNode, start_time, finish_time + timedelta(seconds=1))
    if (refCNode and refDNode):
        refC = ea.getData(refCNode, start_time, finish_time + timedelta(seconds=1))
        refD = ea.getData(refDNode, start_time, finish_time + timedelta(seconds=1))
    else:
        refC = None
        refD = None
    SQI = ea.getData(SQINode, start_time, finish_time + timedelta(seconds=1))
    
    if (refC and refD):
        mask = find_reference_mask_nico(input_data, input_dates, refA, refB, refC, refD, SQI)
    else:
        mask = find_reference_mask_opus(input_data, input_dates, refA, refB, SQI)

    if (len(mask) != len(input_data)):
        log.warning("The reference and value streams have different number of data points \n No reference filter applied")
        return input_data
    
    quality = np.zeros(len(input_data))
    quality[mask] = 1

    # filter from mask
    input_data_filtered = input_data.copy()
    input_data_filtered[mask] = np.NAN
    
    # interpolate
    mask = np.isnan(input_data_filtered)
    global MAX_GAP
    mask = mask_nan(mask, MAX_GAP)
   
    if(np.all(mask)):
        log.warning("All data points filtered out by the reference filter")
    else:
        input_data_filtered[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), input_data_filtered[~mask])
    
    return input_data_filtered, quality


def filter_data(source_node, dest_node, refANode, refBNode, refCNode, refDNode, 
                SQINode, qnode, upper_threshold, lower_threshold, changing_rate, 
                src_read_api_key, src_write_api_key, dest_read_api_key, dest_write_api_key):
    """ 
    gets data from eagle io and if there is new data filters it
        and reuploads it.

    If the src_read/write_api_key is defined it is used as the key for the 
    source node, api_read/write_key for the destination. For any not set, 
    the default is the p25 key (see eagleFilter.py).
    If qnode is defined the quality code is also uploaded to that node 
    """

    ea_src = eagle(read_key=src_read_api_key, write_key=src_write_api_key)
    ea_dest = eagle(quality_node=qnode, read_key=dest_read_api_key, write_key=dest_write_api_key)

    global FILTER_MIN_WINDOW

    source_metadata = ea_src.getLocationMetadata(source_node)
    dest_metadata = ea_dest.getLocationMetadata(dest_node)
    if(source_metadata == -1 or dest_metadata == -1):
        return 0

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
    #dest_metadata['currentTime'] = source_metadata['oldestTime']
    ### for testing when already filtered
    dest_metadata['currentTime'] = source_metadata['currentTime'] - timedelta(days=5)

    # check that there is new data
    if dest_metadata['currentTime'] < source_metadata['currentTime']:
        # get all new data plus a delay window
        start_time =  dest_metadata['currentTime'] - timedelta(days=FILTER_MIN_WINDOW)
        finish_time = source_metadata['currentTime']
       
        # the get in the eagle api is not inclusive so add one second to 
        # finish_time so that all the data including the last point is retrieved 
        # and the process will not be repeated 
        data = ea_src.getData(source_node, start_time, finish_time + timedelta(seconds=1)) 
        if (data == -1):
            return 0
      
        # format data
        input_data = np.asarray(data)[:,1]   
        input_data = input_data.astype(float)

        input_dates = np.asarray(data)[:,0]  
        start_time = datetime.strptime(input_dates[0], '%Y-%m-%dT%H:%M:%S')
        finish_time = datetime.strptime(input_dates[-1], '%Y-%m-%dT%H:%M:%S')
        
        # all new data is nan so no filtering occurs
        if(np.isnan(input_data).all()):
            return 0

        # run all realtime filters
        filtered_data, quality = run_filter(input_data, upper_threshold, 
                lower_threshold, changing_rate, start_time, finish_time, 
                refANode, refBNode, refCNode, refDNode, SQINode, input_dates, 
                src_read_api_key)

        # Create JTS JSON time series of filtered data  
        ts = ea_src.createTimeSeriesQualityJSON(data,filtered_data, quality)

        # update destination on Eagle with filtered data
        res = ea_dest.updateData(dest_node, ts)

        return 1

    log.warning("No new data, no filtering occurred")
    return 0

def main(event, context):
    log.basicConfig(level=log.INFO)

    ''' only required for sns '''
    
    payload = event['Records'][0]['Sns']['Message']
    event = json.loads(payload)
    
    ''' '''

    source_node = event['source']
    dest_node = event['destination']
    upper_threshold = float(event['upperThreshold'])
    lower_threshold = float(event['lowerThreshold'])
    changing_rate = float(event['changingRate'])

    try:
        src_read_api_key = event['sourceReadApiKey']
    except:
        src_read_api_key = ""

    try:
        src_write_api_key = event['sourceWriteApiKey']
    except:
        src_write_api_key = ""

    try:
        dest_read_api_key = event['destReadApiKey']
    except:
        dest_read_api_key = ""

    try:
        dest_write_api_key = event['destWriteApiKey']
    except:
        dest_write_api_key = ""

    try:
        sensor_type = event['sensor']
    except:
        sensor_type = None 
        refANode = None
        refBNode = None
        refCNode = None
        refDNode = None
        SQINode = None
    
    try:
        qnode = event['qnode']
    except:
        qnode = None

    if(sensor_type == 'nico'):
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
    elif(sensor_type == 'opus'):
        try:
            refANode = event['abs360Node']
            refBNode = event['abs210Node']
            refCNode = None
            refDNode = None
            SQINode = event['SQINode']
        except:
            refANode = None
            refBNode = None
            refCNode = None
            refDNode = None
            SQINode = None

    # may need to stay print
    log.info("Processing %s %s " % (source_node, dest_node))
    res = filter_data(source_node, dest_node, 
            refANode, refBNode, refCNode, refDNode, SQINode, qnode,
            upper_threshold, lower_threshold, changing_rate, src_read_api_key, 
            src_write_api_key, dest_read_api_key, dest_write_api_key)
    if(res == 1):
        response = {
                "statusCode": 200,
                "body": ''
            }
    
        return response

######

def run_on_all():
    """ 
    Run the filter on all of the nodes in filterSettings.json locally
    Used for testing and rerunning filter on all nodes when a change is made
    """
    f = open("filterSettings.json", 'r')
    file_contents = json.load(f)
    nodes = file_contents["nodes"]

    # For each node create a dictionary with the same format as an SNS message
    # (I coppied a sample SNS message from the output of the lambda) 
    # The "Message" field is the payload, one entry in filterSettings when running 
    # from gbrEagleFilterMaster.
    for i in range(0, len(nodes)):
        nodei = json.dumps(nodes[i])
        event = {'Records': [{'EventSource': 'aws:sns', 
                    'EventVersion': '1.0', 'EventSubscriptionArn': 'arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate:1cc5186a-04cc-430a-8065-fa438521d082', 'Sns': {'Type': 'Notification', 'MessageId': 'bc85683f-2efc-50c6-8314-3d51aff722d2', 'TopicArn': 'arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate', 
                    'Subject': None, 
                    'Message': '%s'%(nodei), 
                    'Timestamp': '2019-06-03T01:58:35.515Z', 'SignatureVersion': '1', 'Signature': 'MD2dPjKLTGTijU1s+vPuE699sSM7vquQHQFpVBtqECLEX+4psmZeT7oAMSZY5yCAtS2QKesiE4/lR9ezBENfmmTy/TrWyqguyY+4RO121nzlMWN3FN/IPdbNJU2yvsYby7//PwIJDvgN2KgoAhZPoW92bJtFAxOlMKmnNSsfCPM7lH0FF4M2pyvmzbyauFoFhJfdr0hRWfcPnmmMSusr8rc9Y0wdEtR37qexQ99GR8w2KWMZE8VWPNc8ZdXSeE3sLv7floxaxCIqWcS3nm6pJiN/B0YzDBIJvVEIa492qKm8lPd34MCRG6lLH05VJw3KwkOQLbabpJoP43lKhDZdkQ==', 'SigningCertUrl': 'https://sns.ap-southeast-2.amazonaws.com/SimpleNotificationService-6aad65c2f9911b05cd53efda11f913f9.pem', 'UnsubscribeUrl': 'https://sns.ap-southeast-2.amazonaws.com/?Action=Unsubscribe&SubscriptionArn=arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate:1cc5186a-04cc-430a-8065-fa438521d082', 'MessageAttributes': {}}}]}
        main(event, None)
    # Making too many requests can cause eagle to not accept more, so if the 
    # connection is forcibly closed wait a few seconds and try a subsection of nodes
    return

if __name__ == "__main__":
	run_on_all()

    # # For running on/testing one sns message
    # # output on column 3
    # test = {'Records': [{'EventSource': 'aws:sns', 
    #             'EventVersion': '1.0', 'EventSubscriptionArn': 'arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate:1cc5186a-04cc-430a-8065-fa438521d082', 'Sns': {'Type': 'Notification', 'MessageId': 'bc85683f-2efc-50c6-8314-3d51aff722d2', 'TopicArn': 'arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate', 
    #             'Subject': None, 
    #             'Message': '{"name": "p25 - behana creek - ROM-NO3-N", "refANode": "5c3578fc1bbcf10f7880ca62", "refBNode": "5c3578fc1bbcf10f7880ca63", "refCNode": "5c3578fc1bbcf10f7880ca64", "refDNode": "5c3578fc1bbcf10f7880ca65", "SQINode": "5c3578fc1bbcf10f7880ca61","sensor": "nico", "source":"5c3578fc1bbcf10f7880ca5f", "destination":"5ca2a9604c52c40f17064db0", "upperThreshold": "2", "lowerThreshold": "0", "changingRate": "0.05"}', 
    #             'Timestamp': '2019-06-03T01:58:35.515Z', 'SignatureVersion': '1', 'Signature': 'MD2dPjKLTGTijU1s+vPuE699sSM7vquQHQFpVBtqECLEX+4psmZeT7oAMSZY5yCAtS2QKesiE4/lR9ezBENfmmTy/TrWyqguyY+4RO121nzlMWN3FN/IPdbNJU2yvsYby7//PwIJDvgN2KgoAhZPoW92bJtFAxOlMKmnNSsfCPM7lH0FF4M2pyvmzbyauFoFhJfdr0hRWfcPnmmMSusr8rc9Y0wdEtR37qexQ99GR8w2KWMZE8VWPNc8ZdXSeE3sLv7floxaxCIqWcS3nm6pJiN/B0YzDBIJvVEIa492qKm8lPd34MCRG6lLH05VJw3KwkOQLbabpJoP43lKhDZdkQ==', 'SigningCertUrl': 'https://sns.ap-southeast-2.amazonaws.com/SimpleNotificationService-6aad65c2f9911b05cd53efda11f913f9.pem', 'UnsubscribeUrl': 'https://sns.ap-southeast-2.amazonaws.com/?Action=Unsubscribe&SubscriptionArn=arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate:1cc5186a-04cc-430a-8065-fa438521d082', 'MessageAttributes': {}}}]}
    
    # main(test, None)


	



import numpy as np
from datetime import timezone
from datetime import datetime
from datetime import timedelta
from eagleFilter import eagleFilter as eagle
from time import sleep
import json
import io

#Historic eagle io url API
HISTORIC_API = 'https://api.eagle.io/api/v1/historic'

#Eagle io API to get station data
data_api = 'https://api.eagle.io/api/v1/nodes/'

MAX_RECORDS = 10000
INTERVAL = 96  #hours

#Time in AEST
START_TIME = '2019-01-01T00:00:00Z' #in AEST
FINISH_DATE = '2019-04-16T00:00:00Z' #in AEST

datetime_start_time = datetime.strptime(START_TIME, '%Y-%m-%dT%H:%M:%SZ')
datetime_finish_time = datetime.strptime(FINISH_DATE, '%Y-%m-%dT%H:%M:%SZ')

UTC_TIMEZONE = timezone(timedelta(hours=10))
EAGLE_TZ_STRING = 'Australia/Brisbane'

#AWS Configuration
# Credentails are stored in .aws/credentials
#AWS S3 storage default bucket and folder informtaion
BUCKET_NAME = 'digiscapegbr'
DIRECTORY = 'filterdata'


def main(event, context):
    # Create a new eagle utility instancee
    ea = eagle()

    # # get node configuration from AWS
    # nodes = ea.getFileAWSJSON('digiscapegbr', 'filterdata', 'filterNodesConfig.json')

    # # for each node to be filtered
    # for node in nodes['nodes']:
    #     sourceNode = node['source']
    #     destNode = node['destination']
    
    #For testing set up source and destination nodes
    sourceNode = '5b177a5de4b05e726c7eeecc'
    destNode = '5ca2a9604c52c40f17064daf'


    # Grab the metadata
    sourceMetadata = ea.getLocationMetadata(sourceNode)
    destMetadata = ea.getLocationMetadata(destNode)
    
    # print('~~~~~~')
    # print(sourceMetadata['currentTime']- timedelta(hours=2))
    # print(destMetadata['currentTime'])
    
    # sourceMetadata['currentTime'] = sourceMetadata['currentTime']- timedelta(hours=2)

    # # How are we going to make sure the vectors for filtering are big enough
    while destMetadata['currentTime'] < sourceMetadata['currentTime']:
        # print(destMetadata['currentTime'])
        # print('time:')
        # print(sourceMetadata['currentTime'])

        # get the last time form destination as the startTime
        startTime = destMetadata['currentTime']
        # Add five minutes to make sure we get the latest value
        #finishTime = sourceMetadata['currentTime'] - timedelta(hours=1) + timedelta(minutes=5)
        finishTime = sourceMetadata['currentTime']
        # Get data from eagle
        # print('!!!!')
        # print(startTime)
        # print(finishTime)
        data = ea.getData(sourceNode, startTime, finishTime)
            # data = ea.getData(sourceNode, datetime_start_time, datetime_finish_time)

        # filter data - data is an array of time,value pairs
        # f = filter(data)

    
        # prepare data format, convert to pandas dataframe
        input_data = np.asarray(data)[:,1]
        input_time = np.asarray(data)[:,0]
        
        # for i in input_time:
        #     i = datetime.strptime(i, '%Y-%m-%dT%H:%M:%S')
        
        
        input_data = input_data.astype(float)
        
    
        
        # filter settings for threshold and changing rate
        Upper_Threshold = 2
        Lower_Threshold = 0
        Changing_Rate = 0.5
        
        
        # apply threshold filter
        # filtered data is replaced by NAN
        mask = (input_data<Lower_Threshold)|(input_data>Upper_Threshold)
        input_data_filtered = input_data.copy()
        input_data_filtered[mask] = np.NAN
        
        
        
        # interpolate
        # linear interpolation to remove NAN
        mask = np.isnan(input_data_filtered)
        input_data_filtered[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), input_data_filtered[~mask])
    
    
        
        # Changing rate filter
        # filtered data is replaced by NAN
        diff_index = np.diff(input_data_filtered)
        
        mask_diff = (abs(diff_index)>Changing_Rate)
        mask_diff = np.insert(mask_diff, 0, False)
        
        input_data_filtered_seocnd = input_data_filtered.copy()
        input_data_filtered_seocnd[mask_diff] = np.NAN
        
        # interpolate
        # linear interpolation to remove NAN
        mask = np.isnan(input_data_filtered_seocnd)
        input_data_filtered_seocnd[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), input_data_filtered_seocnd[~mask])
    
        
        
        
        
        # save_df = df.copy()
        # save_df.index = input_time
        # save_df.index = pd.to_datetime(save_df.index)
        
        
        
        
        
        
        # save_df.to_csv('filtered.csv',header=None)
        # save_origin_df.to_csv('origin.csv',header=None)
        
        
        # convert pandas data frame to list
        f = input_data_filtered_seocnd.tolist()
    
        
    
        
        #Create JTS JSON time series
        ts = ea.createTimeSeriesJSON(data,f)
            
            
    
        # # pay attention to the data types
        # fa = np.asarray(data)[:, 1]
        # fa = np.float64(fa)
        # # Create JTS JSON time series
        # ts = ea.createTimeSeriesJSON(data, fa)
    
        # Update Eagle
        res = ea.updateData(destNode, ts)
        #
        # print(res)
    
        # give a little time for ingest on eagle to happen
        #sleep(0.2)
    
        # update metadata
        sourceMetadata = ea.getLocationMetadata(sourceNode)
        destMetadata = ea.getLocationMetadata(destNode)
    
    
    
    
        response = {
            "statusCode": 200,
            "body": ''
        }
    
        return response
        
        
        




if __name__ == "__main__":
    main('', '')



'''
Filters the provided node by provided parameters
Used in gbrEagleFilterNode
'''

import numpy as np
from eagleFilter import eagleFilter as eagle
import json

"""Function that takes the node, loads data and runs both filters on it"""
def run_filter(input_data, upper_threhold, lower_threhold, changing_rate):
    # Directly from lambda_function.py
    # apply threshold filter
    # filtered data is replaced by NAN
    mask = (input_data<lower_threhold)|(input_data>upper_threhold)
    input_data_filtered = input_data.copy()
    input_data_filtered[mask] = np.NAN
    
    # interpolate
    # linear interpolation to remove NAN
    mask = np.isnan(input_data_filtered)
    input_data_filtered[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), input_data_filtered[~mask])

    # Changing rate filter
    # filtered data is replaced by NAN
    diff_index = np.diff(input_data_filtered)
    
    mask_diff = (abs(diff_index)>changing_rate)
    mask_diff = np.insert(mask_diff, 0, False)
    
    input_data_filtered_seocnd = input_data_filtered.copy()
    input_data_filtered_seocnd[mask_diff] = np.NAN
    
    # interpolate
    # linear interpolation to remove NAN
    mask = np.isnan(input_data_filtered_seocnd)
    input_data_filtered_seocnd[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), input_data_filtered_seocnd[~mask])
    

    # may be unneccesary 
    # convert pandas data frame to list
    f = input_data_filtered_seocnd.tolist()

    return f

""" function that gets data from eagle io and if there is new data filters it
and reuloads it 

Could have this in run filter or in the filter loop but not sure where so 
having seperate
"""
def filter_data(source_node, dest_node, upper_threhold, lower_threhold, changing_rate):
    # get input data from node and convert
    ea = eagle()

    source_metadata = ea.getLocationMetadata(source_node)
    dest_metadata = ea.getLocationMetadata(dest_node)
    # what do these look like?
    print(source_metadata)

    # check that there is new data
    if dest_metadata['currentTime'] < source_metadata['currentTime']:
        # get all new data
        start_time = dest_metadata['currentTime']
        finish_time = source_metadata['currentTime']
        data = ea.getData(source_node, start_time, finish_time)

        # format data
        input_data = np.asarray(data)[:,1]     
        input_data = input_data.astype(float)

        # run all realtime filters
        filtered_data = run_filter(input_data, upper_threhold, lower_threhold, changing_rate)
        # Create JTS JSON time series of filtered data       
        ts = ea.createTimeSeriesJSON(data,filtered_data)
        # update destination on Eagle with filtered data
        res = ea.updateData(dest_node, ts)

        return 1
    return 0

def main(event, context):
    source_node = event['source_node']
    dest_node = event['dest_node']
    upper_threshold = event['upper_threshold']
    lower_threshold = event['lower_threshold']
    changing_rate = event['changing_rate']

    print(source_node, dest_node)
    res = filter_data(source_node, dest_node, upper_threshold, 
            lower_threshold, changing_rate)
    if(res == 1):
        response = {
                "statusCode": 200,
                "body": ''
            }
    
        return response


if __name__ == "__main__":
    main()

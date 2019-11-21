"""
Lambda script to determine thresholds for nodes from percentiles over the last 
6 months and update settings json file accordingly
"""
import numpy as np
from eagleFilter import eagleFilter as eagle
import json
import logging as log
from datetime import timedelta

def main(event, context):
    log.basicConfig(level=log.ERROR)
      
    # get all node from S3
    ea = eagle()
    try:
        settings = ea.getFileAWSJSON('digiscapegbr', 'filterSettings', 'filterSettings.json')
    except Exception as e:
        log.error("There was an error in importing the node settings file.")
        print(e)
        return
        
    nodes = settings["nodes"]
    for node in nodes:
        source = node['source']
        name = node["name"]
        try:
            api_key = node['sourceReadApiKey']
            ea.setApiReadKey(api_key)
        except:
            pass

        # Only want to modify thresholds for nitrate and water levels so
        # search for these in the name and set flags to test for later
        level = nitrate = 0
        if("level" in name.lower()):
            level = 1
        if(("no3" in name.lower()) or ("nitrate" in name.lower())):
            nitrate = 1
            
        # if this is neither a waterlevel nor nitrate node no changes required
        # move onto the next node immediately
        if (level == 0 and nitrate == 0):
            continue

        source_metadata = ea.getLocationMetadata(source)
        if (source_metadata) == -1:
            continue
        if (source_metadata['currentTime']==0):
            # potentially empty node or some other issue
            continue
        # ? do we want from olderst time or last month or last year ext
        print(source_metadata['currentTime'] )
        historical_data = ea.getData(source, source_metadata['currentTime'] - timedelta(days=178), source_metadata['currentTime'])
        if (historical_data) == -1:
            continue

        # seperate out values in node from dates
        values = np.asarray(historical_data)[:,1].astype(float)

        if(nitrate):
            upper_threshold = np.percentile(values, 95)
            node['upperThreshold'] = str(upper_threshold)

        if(nitrate or level): # should always test true
            rate_of_change = abs(np.diff(values))
            rate_threshold = np.percentile(rate_of_change, 95)
            node['changingRate'] = str(rate_threshold)

    ea.uploadDirectAWSJSON('digiscapegbr', 'filterSettings', 'filterSettings.json', settings)
    return 0

if __name__ == "__main__":
    main(None, None)
 
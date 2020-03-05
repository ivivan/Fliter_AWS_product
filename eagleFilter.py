"""
version 4
"""

import boto3
import botocore
import botocore.vendored.requests as requests
import math
from botocore.vendored.requests.exceptions import ConnectionError
import json
import io
from datetime import datetime, timedelta
import logging 
# import requests
""" 
there is a depreciation warning for the use of botocore requests but as this comes
in a layer provided by aws it is left to be
"""

# Set up logger
log = logging.getLogger()
log.setLevel(logging.WARNING)
p25apikey = 'GBBbwpSHH54zF58e7Xwp25zFUZ8xJ5c3TxHUff1B'

"""
Class that wraps functioanlity for interacting with eagle.io
Instead of throwing errors this class will output them to the log, so that 
    repetition in lambda for error conditions is avoided
"""
class eagleFilter():
    # Historic eagle io url API
    HISTORIC_API = 'https://api.eagle.io/api/v1/historic'

    # Eagle io API to get station data
    data_api = 'https://api.eagle.io/api/v1/nodes/'

    # AWS Configuration
    # Credentails are stored in .aws/credentials
    # AWS S3 storage default bucket and folder informtaion
    BUCKET_NAME = 'digiscapegbr'
    DIRECTORY = 'filterdata'

    # api key set to p25 api for backward compatibility, will be overriden if a new api is passed in
    api_read_key = 'GBBbwpSHH54zF58e7Xwp25zFUZ8xJ5c3TxHUff1B'
    api_write_key = 'lKTpXokuT0Plrin0GakpbSa1fKeftTP5Lk5rZeVo'
    qnode = ""

    # maximum data points to load from eagle
    # too low to run once on some old streams on when filtering for the first time
    load_limit = 5000 

    # Set up logger
    # log.basicConfig(level=log.WARNING)

    def __init__(self, read_key=None, write_key = None, quality_node=None):
        """
        read_key/write_key: API read/write key for communication with eagle, 
            if not set by default the p25 key will be used 
            (nodes will need to be in p25 workspace)
        quality_node: if the quality_node is set the quality code will be
            uploaded to this node whenever data is updated, as well as being 
            attached to the primary node given to updateData. 
        """
        if(read_key):
            self.api_read_key = read_key
        if(write_key):
            self.api_write_key = write_key
        if(quality_node):
            self.qnode = quality_node

    def setApiReadKey(self, key):
        self.api_read_key = key

    def setApiWriteKey(self, key):
        self.api_write_key = key

    def setLoadLimit(self, limit):
        """ Set the maximum number of data points to request from eagle """
        self.load_limit = limit

    def uploadDataAWSJSON(self, bucket_name, directory, filename):
        """ Upload json file 'filename' to the S3 directory specified by 'bucket_name'
            and 'directory'.
        """

        client = boto3.resource(
            's3',
            region_name='ap-southeast-2',
            aws_access_key_id='',
            aws_secret_access_key='cDrq6D//PsrA0QkFBfW2CezUt2LDgRL+lyQApsEa',
        )

        fname = open(filename, 'r')
        memio = io.BytesIO(fname.read().encode('utf-8'))
        fname.close()

        # note that the content type needs to be changed depending on file type
        client.Bucket(bucket_name).put_object(
            Key= directory +'/'+filename,
            Body=memio,
            ContentType='application/json')
        memio.close()
    
    def uploadDirectAWSJSON(self, bucket_name, directory, filename, data):
        """ Upload dictionary data to the S3 directory bucket_name/directory
        under the name 'filename'
        """
        client = boto3.resource(
            's3',
            region_name='ap-southeast-2',
            aws_access_key_id='',
            aws_secret_access_key='cDrq6D//PsrA0QkFBfW2CezUt2LDgRL+lyQApsEa',

        )
        client.Object(bucket_name, directory+ '/' +filename).put(
             Body=(bytes(json.dumps(data, indent=2).encode('UTF-8')))
        )

    def getFileAWSJSON(self,bucket_name, directory, filename):
        """
        get json file from AWS S3 and return its deserailised contents (dictionary)
        """
        # Start is datetime, interval in hours
        # Check whether a bucket exist
        # This only works for json files
        fileContents = ''

        client = boto3.client(
            's3',
            region_name='ap-southeast-2',
            aws_access_key_id='',
            aws_secret_access_key='cDrq6D//PsrA0QkFBfW2CezUt2LDgRL+lyQApsEa',
        )

        try:
            key_name = directory + '/' + filename
            results = client.list_objects(Bucket=bucket_name, Prefix=key_name)

            if 'Contents' in results:
                obj = client.get_object(Bucket=bucket_name, Key=key_name)
                fileContents = json.loads(obj['Body'].read())
                #dt = datetime.strptime(j['lasttime'], "%Y%m%d%H%M")

        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
        # returns an in memory file
        return fileContents

    def JTStoArray(self,ts):
        #helper function to convert JTS to an array of time value pairs
        data = []

        for i in range(0, len(ts['data'])):
            # make sure to check for NaN
                # Format for data array
                row = [ts['data'][i]['ts'] + 'Z', ts['data'][i]['f']['0']['v']]
                data.append(row)

        return data

    # This function is the base funtion to retreive metadata from Eagle.
    def getLocationMetadata(self, node):
        """
        Retreive metadatd from Eagle. Metadata is filtered to and significant dates are returned

        :param node: node Id of the target node
        :return: dictionary with values for 'oldestTime', 'currentTime', 'obsInterval', 'previousTime', and
                'lastFiltered'. lastFiltered and currentTime may be 0 if there is no previous data. Values will be None
                if any cannot be parsed as dates.
        """

        headers = {'X-Api-Key': self.api_read_key}

        jsonData = requests.get(self.data_api+node, headers=headers).json()
        if(jsonData.get('error') != None):
            log.error("Error %s for node %s in loading Eagle stream metadata: %s." %
                        (jsonData['error'].get('code'),
                         node,
                        jsonData['error'].get('message')))
            return -1

        try:
            metadata = jsonData["metadata"]
            # the metadata is a list of dictionaries and we want the dictionary where the name key has value
            # lastFiltered. Result of the lambda: [{'name' : 'lastFiltered', 'value': ...}]
            lastFilteredMetadata = list(filter(lambda metadata: metadata['name'] == 'lastFiltered', metadata))[0]
            lastFiltered = datetime.strptime(lastFilteredMetadata['value'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(microsecond=0)
        except:
            lastFiltered = 0

        try:
            currentTime = datetime.strptime(jsonData['currentTime'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(microsecond=0)
            prevTime = datetime.strptime(jsonData['previousTime'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(microsecond=0)
            oldestTime = datetime.strptime(jsonData['oldestTime'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(microsecond=0)
            obsInterval = currentTime - prevTime
        except:
            currentTime = 0 # no previous data
            prevTime = None
            oldestTime = None
            obsInterval = None

        returnData = { 'oldestTime':oldestTime, 'currentTime':currentTime,
                       'obsInterval': obsInterval, 'previousTime':prevTime, 'lastFiltered' : lastFiltered}

        return returnData

    def getMetadata(self, node):
        """
        Base funtion to retreive metadatd from Eagle.

        :param node: node Id for target node
        :return: result from request or -1 if there is an error
        """

        headers = {'X-Api-Key': self.api_read_key}

        jsonData = requests.get(self.data_api + node, headers=headers).json()
        if (jsonData.get('error') != None):
            log.error("Error %s in loading Eagle stream metadata: %s." %
                      (jsonData['error'].get('code'),
                       jsonData['error'].get('message')))
            return -1

        return jsonData


    # this function is the base funtion to retreive current and prevtime from Eagle.
    def getcurrentPrevTime(self, node):
        headers = {'X-Api-Key': self.api_read_key}

        jsonData = requests.get(self.data_api + node, headers=headers).json()
        if(jsonData.get('error') != None):
            raise(Exception("Error %s in loading Eagle stream data: %s." % 
                        (jsonData['error'].get('code'), 
                        jsonData['error'].get('message'))))

        currentTime = jsonData['currentTime']
        prevTime = jsonData['previousTime']

        return currentTime, prevTime

    # Function to return data for node from eagle io
    def getData(self, node, startTime, endTime):
        """ Retireve data from eagle.io node between startTime and endTime """
        headers = {'X-Api-Key':self.api_read_key}
        data = []

        #Get utc representation
        params = {'startTime':startTime.strftime('%Y-%m-%dT%H:%M:%SZ'), 
                  'endTime':endTime.strftime('%Y-%m-%dT%H:%M:%SZ'),
                  'timeFormat' : 'YYYY-MM-DDTHH:mm:ss',
                  'timezone' : 'Etc/UTC',
                  'limit': self.load_limit, 
                  'params':node }
        try:
            #could be in a repeat loop
            for i in range(0,3):
                hist_values = requests.get(self.HISTORIC_API, headers=headers, params=params).json()
                if len(hist_values) > 0:
                    if(hist_values.get('error') != None):
                        log.error("Error %s in loading Eagle historical data: %s." % 
                                (hist_values['error'].get('code'), 
                                hist_values['error'].get('message')))
                        return -1
                    break

        except ConnectionError as e:
            log.error(e)

        if len(hist_values['data']) > 0:
            rows = []
            for i in range(0, len(hist_values['data'])):
                ts = hist_values['data'][i]['ts']
                if '0' in hist_values['data'][i]['f'] and 'v' in hist_values['data'][i]['f']['0']:
                    val = hist_values['data'][i]['f']['0']['v']
                else:
                    val = 0
                data.append([ts, val])
      
        if (endTime - datetime.strptime(data[-1][0], '%Y-%m-%dT%H:%M:%S') > timedelta(hours=1)):
            log.warning("Not all requested dates were returned, data may not " 
                        + "extend to the requested end date, or the set limit "
                        + str(self.load_limit) + "may have been exceeded."
                        + "\n requested end: " + str(endTime) 
                        + " returned end: " + data[-1][0])

        return data


    def createTimeSeriesJSON(self, origData, filteredData):
        """ Create a json timeseries for upload """
        data = []

        for i in range(0,len(origData)):
            # make sure to check for NaN
            if not math.isnan(filteredData[i] ):
                # Format in JTS
                v = {"ts": origData[i][0]+'Z', "f":{"0":{"v": filteredData[i]}}} 
                data.append(v)
        # Construct timeseries for loading
        ts = {"docType": "jts",
              "version": "1.0",
              "data":data}
        return json.dumps(ts)

    # Alternatively this routine constructs a csv to be stored on S3
    def createTimeSeriesCSV(self, origData, filteredData):
        """ Create a csv file from two arrays, the original and filtered values """

        data = 'Date,FilteredValue\r'
        for i in range(0,len(origData)):
            if not math.isnan(filteredData[i]):
                data = data + '%s,%.2f\r' % (origData[i][0], filteredData[i])

        return data

    # This function creates the json timeseries for upload, including a quality code column
    def createTimeSeriesQualityJSON(self, origData,filteredData, quality):
        """ Create a json timeseries for upload including a data column (0) with
        quality code and a quality code column (1)        
        """
        data = []

        for i in range(0,len(origData)):
            # make sure to check for NaN
            if not math.isnan(filteredData[i] ):
                # Format in JTS
                v = {"ts": origData[i][0]+'Z', 
                "f":{
                    "0":{"v": filteredData[i], "q":quality[i]},
                    "1":{"v":quality[i]}
                    }
                } 
                data.append(v)
    
        #Construct timeseries for loading
        ts = {"docType": "jts",
              "version": "1.0",
              "data":data}
        return json.dumps(ts)

    # Update Eagle
    def updateData(self, node, data):
        """ Upload data to eagle.io node 
        node : node id for target node
        data: time series for upload
        """
        # make sure to use write API key
        headers = {'X-Api-Key': self.api_write_key, 'Content-Type':'application/json'}
    
        params = {'params': node + '(columnIndex:0)'+ "," + self.qnode + '(columnIndex:1)'}

        try:
            result = requests.put(self.HISTORIC_API, data=data, headers=headers, params=params, timeout=2.0)
            if(result.ok != True):
                log.warning("Data could not be uploaded, put returned with error code %s: %s.", 
                    json.loads(result.text).get('error').get('code'), 
                    json.loads(result.text).get('error').get('message'))

            log.info(result)
         
        except ConnectionError as e:
            # exception is printed not raised so that the function is not repeated for this
            result = -1
            print(e)
            #raise

        return result
    

    # Update Eagle metadata
    def updateMetadata(self, metadata):
        #cannot update Location
        #make sure to use write API key
        headers = {'X-Api-Key': self.api_write_key, 'Content-Type': 'application/json'}

        try:
            result = requests.put(self.data_api, data=metadata, headers=headers, timeout=2.0)
            if(result.ok != True):
                log.warning("Metadata could not be uploaded, put returned with error code %s: %s.",
                    json.loads(result.text).get('error').get('code'),
                    json.loads(result.text).get('error').get('message'))

        except ConnectionError as e:
            result = -1
            print(e)
            #raise

        return result

    #
    def updateLocationMetadata(self, node, metadata):
        """
        Update Eagle Metadata for a particular Node

        :param node: target node
        :param metadata: json string with the parentId and metadata as a list of dictionaries
                            {"parentId" : "....",
                             "metadata": [{"name": "...", "value": "..."}, {"name": "...", "value": "..."}}
        :return: response from eagle.io or -1 if there is a connection error
        """
        #cannot update Location
        #make sure to use write API key
        headers = {'X-Api-Key': self.api_write_key,'Content-Type':'application/json'}

        try:
            result = requests.put(self.data_api + node, data=metadata, headers=headers, timeout=2.0)
            if(result.ok != True):
                log.warning("Metadata could not be uploaded, put returned with error code %s: %s.",
                    json.loads(result.text).get('error').get('code'),
                    json.loads(result.text).get('error').get('message'))
        except ConnectionError as e:
            result = -1
            print(e)
            #raise

        return result



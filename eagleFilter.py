"""
version 3.0
"""

import boto3
import botocore
import botocore.vendored.requests as requests
import math
from botocore.vendored.requests.exceptions import ConnectionError
import json
import io
from datetime import datetime
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
    api_key = 'GBBbwpSHH54zF58e7Xwp25zFUZ8xJ5c3TxHUff1B'
    qnode = ""

    # maximum data points to load from eagle
    # too low to run once on some old streams on when filtering for the first time
    load_limit = 5000 

    # Set up logger
    # log.basicConfig(level=log.WARNING)

    def __init__(self, key=None, quality_node=None):
        if(key):
            self.api_key = key
        if(quality_node):
            self.qnode = quality_node

    def setApiKey(self, key):
        self.api_key = key

    def setLoadLimit(self, limit):
        """ Set the maximum number of data points to request from eagle """
        self.load_limit = limit

    def uploadDataAWSJSON(self, bucket_name, directory, filename):
        """ Upload file 'filename' to the S3 directory specified by 'bucket_name'
            and 'directory'.
        """

        client = boto3.resource(
            's3',
            region_name='ap-southeast-2',
            aws_access_key_id='AKIAV7HZ4YHAKWC4NAHG',
            aws_secret_access_key='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',
        )

        fname = open(filename, 'r')
        memio = io.BytesIO(fname.read().encode('utf-8'))
        fname.close()

        # note that the content type needs to be changed depenging on file type
        client.Bucket(bucket_name).put_object(
            Key= directory +'/'+filename,
            Body=memio,
            ContentType='application/json')
        memio.close()
    
    def uploadDirectAWSJSON(self, bucket_name, directory, filename, data):
        """ Upload dictionary data to the S3 directory bucket_name/directory
        under the name filename 
        """
        client = boto3.resource(
            's3',
            region_name='ap-southeast-2',
            aws_access_key_id='AKIAV7HZ4YHAKWC4NAHG',
            aws_secret_access_key='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',

        )
        client.Object(bucket_name, directory+ '/' +filename).put(
             Body=(bytes(json.dumps(data, indent=2).encode('UTF-8')))
        )

    def getFileAWSJSON(self,bucket_name, directory, filename):
        # Start is datetime, interval in hours
        # Check whether a bucket exist
        # This only works for json files
        fileContents = ''

        client = boto3.client(
            's3',
            region_name='ap-southeast-2',
            aws_access_key_id='AKIAV7HZ4YHAKWC4NAHG',
            aws_secret_access_key='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',
        )

        try:
            key_name = directory +'/' + filename
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
        #returns an in memory file
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


    #this function is the base funtion to retreive metadat from Eagle.
    def getLocationMetadata(self, node):
        headers = {'X-Api-Key': self.api_key}


        jsonData = requests.get(self.data_api+node, headers=headers).json()
        if(jsonData.get('error') != None):
            log.error("Error %s in loading Eagle stream metadata: %s." % 
                        (jsonData['error'].get('code'), 
                        jsonData['error'].get('message')))
            return -1

        #print("Time %s value %s" %(jsonData.get('currentTime'), jsonData.get('currentValue')))
        # print("###########", jsonData['currentTime'])

        try:
            currentTime = datetime.strptime(jsonData['currentTime'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(microsecond=0)
            prevTime = datetime.strptime(jsonData['previousTime'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(microsecond=0)
            oldestTime = datetime.strptime(jsonData['oldestTime'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(microsecond=0)
            obsInterval = currentTime - prevTime
        except:
            
            currentTime = 0
            prevTime = None
            oldestTime = None
            obsInterval = None

        returnData = { 'oldestTime':oldestTime, 'currentTime':currentTime,
                       'obsInterval': obsInterval, 'previousTime':prevTime}

        return returnData

    # this function is the base funtion to retreive current and prevvtime from Eagle.
    def getcurrentPrevTime(self, node):
        headers = {'X-Api-Key': self.api_key}

        jsonData = requests.get(self.data_api + node, headers=headers).json()
        if(jsonData.get('error') != None):
            raise(Exception("Error %s in loading Eagle stream data: %s." % 
                        (jsonData['error'].get('code'), 
                        jsonData['error'].get('message'))))

        currentTime = jsonData['currentTime']
        prevTime = jsonData['previousTime']

        return currentTime, prevTime

    #Function to return data for node from eagle io
    def getData(self, node, startTime, endTime):
        #using a read API key
        # print(startTime, endTime)

        headers = {'X-Api-Key':self.api_key}
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

        except ConnectionError as e :
            log.error(e)
            # need to do something else here

        # print(hist_values)
        if len(hist_values['data']) > 0:
            rows = []
            for i in range(0,len(hist_values['data']) ):
                ts = hist_values['data'][i]['ts']
                if '0' in hist_values['data'][i]['f'] and 'v' in hist_values['data'][i]['f']['0']:
                    val = hist_values['data'][i]['f']['0']['v']
                else:
                    val = 0
                data.append([ts,val])
            #data.append({'node':node, 'data':rows})

        return data

    #This function creates the json timeseries for upload
    def createTimeSeriesJSON(self, origData,filteredData):
        data = []

        for i in range(0,len(origData)):
            #make sure to check for NaN
            if not math.isnan(filteredData[i] ):
                #Format in JTS
                v = {"ts": origData[i][0]+'Z', "f":{"0":{"v": filteredData[i]}}} # add "q": quality[i]
                data.append(v)
        #Construct timeseries for loading
        ts = {"docType": "jts",
              "version": "1.0",
              "data":data}
        return json.dumps(ts)

    #Alternatively this routine constructs a csv to be stored on S3
    def createTimeSeriesCSV(self,origData,filteredData):

        data = 'Date,FilteredValue\r'
        for i in range(0,len(origData)):
            if not math.isnan(filteredData[i] ):
                data = data + '%s,%.2f\r' % (origData[i][0], filteredData[i])

        return data

    #This function creates the json timeseries for upload, including a quality code column
    def createTimeSeriesJSONq(self, origData,filteredData, quality):
        data = []

        for i in range(0,len(origData)):
            #make sure to check for NaN
            if not math.isnan(filteredData[i] ):
                #Format in JTS
                # v = {"ts": origData[i][0]+'Z', "f":{"0":{"v": filteredData[i], "q":quality[i]}}} # add "q": quality[i]
                
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

    #Update Eagle
    def updateData(self,node,data):
        #make sure to use write API key
        headers = {'X-Api-Key': 'lKTpXokuT0Plrin0GakpbSa1fKeftTP5Lk5rZeVo','Content-Type':'application/json'}
        #node = "5ca2a9604c52c40f17064dbsdfsd0"
    
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
    

    #Update Eagle
    def updateMetadata(self,node,metadata):
        #cannot update Location
        #make sure to use write API key
        headers = {'X-Api-Key': 'lKTpXokuT0Plrin0GakpbSa1fKeftTP5Lk5rZeVo','Content-Type':'application/json'}


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

        return   result



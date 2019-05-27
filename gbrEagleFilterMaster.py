'''
Master filter for intantiating eagleFilter
'''
from eagleFilter import eagleFilter as eagle
import boto3
import json

def main(event, context):
    #get all node from S3
    ea = eagle()
    nodes = ea.getFileAWSJSON('digiscapegbr', 'filterSettings', 'filterSettings.json')["nodes"]

    client = boto3.client('lambda', region_name='ap-southeast-2')
    for node in nodes:
        # set perams, src, dst, threshold, rate
        settings = {}
        settings['source_node'] = node['source']
        settings['dest_node'] = node['destination']
        #convert to nums?
        settings['upper_threshold'] = node['upperThreshold']
        settings['lower_threshold'] = node['lowerThreshold']
        settings['changing_rate'] = node['changingRate']
        # invoke lambda using these 
            # just invokin giflterdata is fine
        print(settings} 
        try: 
            client.invoke(FunctionName='gbrEagleFilterNode', InvocationType='DryRun', Payload=json.dumps(settings))
        except Exception as e:
            print("There was an error in importing the node settings file")
            raise e # FileNotFoundError("There was an error in importing the node settings file")
            '''
            options:
                raise an exception ( can I just raise e)
                let it raise whather exception and log
                -> can't really try anything else here
            '''
            
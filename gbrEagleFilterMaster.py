from eagleFilter import eagleFilter as eagle
import boto3
import json
import logging as log

def main(event, context):
    log.basicConfig(level=log.INFO)
      
    #get all node from S3
    ea = eagle()
    try:
        nodes = ea.getFileAWSJSON('digiscapegbr', 'filterSettings', 'filterSettings.json')["nodes"]
    except Exception as e:
        log.error("There was an error in importing the node settings file.")
        print(e)
        # this shouldn't cause a retry buck check
        return 
        
    #client = boto3.client('lambda', region_name='ap-southeast-2')
    client = boto3.client('sns', region_name='ap-southeast-2')
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
        
        message = json.dumps(settings)
        print(message)
        
        #client.invoke(FunctionName='gbrEagleFilterNode', InvocationType='DryRun', Payload=json.dumps(settings))
        r = client.publish(TopicArn='arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate', Message=message)
        print(r)
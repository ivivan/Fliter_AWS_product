from eagleFilter import eagleFilter as eagle
import boto3
import json
import logging as log

def main(event, context):
    log.basicConfig(level=log.INFO)
      
    # Get all node from S3
    ea = eagle()
    try:
        nodes = ea.getFileAWSJSON('digiscapegbr', 'filterSettings', 'filterSettings.json')["nodes"]
    except Exception as e:
        log.error("There was an error in importing the node settings file.")
        log.error(e)
        # this shouldn't cause a retry but check
        return
        
    client = boto3.client('sns', region_name='ap-southeast-2')
    for node in nodes:
        message = json.dumps(node)
        log.info(message)

        # Invoke directly using:
        # client.invoke(FunctionName='gbrEagleFilterNode', InvocationType='DryRun', Payload=json.dumps(settings))

        r = client.publish(TopicArn='arn:aws:sns:ap-southeast-2:410693452224:gbrNodeUpdate', Message=message)
        log.info("Response: %s", r)

if __name__ == "__main__":
    main(None, None)
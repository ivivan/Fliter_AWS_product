""" 
reads a config and runs filter on it

"""
from filter import filter_data

# lots of variables (such as time) in lambda_function.py were not used and so were excluded


def main(event, context):
    source_node = '5b177a5de4b05e726c7eeecc'
    dest_node = '5ca2a9604c52c40f17064daf'
    upper_threshold = 2
    lower_threshold = 0
    changing_rate = 0.5

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
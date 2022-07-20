import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute difference in readings of log files and write back to logfile')
    parser.add_argument('--log_file_op', type=str, default="logs/log.json",
                        help='input log file path of bn/relu/pool data')
    parser.add_argument('--log_file_conv', type=str, default="logs/log.json",
                        help='input log file path of conv data')

    args = parser.parse_args()

    op_log_file = args.log_file_op
    conv_log_file = args.log_file_conv

    name, ext = op_log_file.split('.')
    op_log_file_with_diffs = name+'_with_diff.'+ext

    with open(op_log_file, 'r') as myfile:
        op_file_data = json.load(myfile)

    conv_data = []
    with open(conv_log_file, 'r') as myfile:
        conv_file_data = json.load(myfile)

    diff_file_data = op_file_data

    for idx, workload in enumerate(op_file_data["workloads"]):
        for slot in range(6):
            op_write = workload['results'][str(slot)]['overall']['write_bytes']
            conv_write = conv_file_data['workloads'][idx]['results'][str(slot)]['overall']['write_bytes']
            write_diff = op_write - conv_write

            op_read = workload['results'][str(slot)]['overall']['read_bytes']
            conv_read = conv_file_data['workloads'][idx]['results'][str(slot)]['overall']['read_bytes']
            read_diff = op_read - conv_read
            diff_file_data['workloads'][idx]['results'][str(slot)]['diff'] = {'write_bytes': write_diff,
                                                                              'read_bytes': read_diff}

    with open(op_log_file_with_diffs,'w+') as myfile:
        json.dump(diff_file_data, myfile, indent=4)
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_df(df, x, y, title="", xlabel='Sample', ylabel='Read B/W', dpi=100):
    plt.figure(figsize=(16, 5), dpi=dpi)
    plt.plot(x, y, color='tab:purple')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

def generate_boxplot(file_data):

    df = pd.DataFrame()
    for item in file_data:
        df0 = pd.DataFrame.from_dict(item["results"]['2']["samples"])
        df0 = df0.drop(columns=["read_bytes","write_bytes","write_bw"])
        df0['workload'] = item["workload_str"]
        df = df.append(df0)

    # boxplot = df.boxplot(column='read_bw', by='workload_str')
    # plt.show()
    df_long = pd.melt(df, "workload", var_name="Read Bandwidth", value_name="Value (MB/s)")
    sns.factorplot("Read Bandwidth", hue="workload", y="Value (MB/s)", data=df_long, kind="box")
    plt.show()


def plot_write_bytes(file_data):

    data_list = []
    for item in file_data:
        data_list.append({"workload":item["workload_str"], "write_bytes":item["results"]['5']["overall"]["write_bytes"]})

    df = pd.DataFrame.from_dict(data_list)

    fig, ax = plt.subplots()
    df.plot('workload','write_bytes',kind='line', ax=ax)
    for k,v in df.iterrows():

        ax.annotate(v.write_bytes,[k,v.write_bytes])
    plt.show()

def plot_read_bytes(file_data):

    data_list = []
    for item in file_data:
        data_list.append({"workload":item["workload_str"], "read_bytes":item["results"]['2']["overall"]["read_bytes"]})

    df = pd.DataFrame.from_dict(data_list)

    fig, ax = plt.subplots()

    df.plot('workload', 'read_bytes', kind='line', ax=ax)
    for k, v in df.iterrows():
        ax.annotate(v.read_bytes, [k, v.read_bytes])
    plt.show()

def plot_bytes(file_data, slot, rOrW):

    data_list = []
    for item in file_data:
        if rOrW == 'read':
            data_list.append({"workload": item["workload_str"], "read_bytes": item["results"][slot]["overall"]["read_bytes"]})
        else:
            data_list.append(
                {"workload": item["workload_str"], "write_bytes": item["results"][slot]["overall"]["write_bytes"]})

    df = pd.DataFrame.from_dict(data_list)

    fig, ax = plt.subplots()

    if rOrW == 'read':
        df.plot('workload', "read_bytes", kind='line', ax=ax)
    else:
        df.plot('workload', "write_bytes", kind='line', ax=ax)

    for k, v in df.iterrows():
        if rOrW == 'read':
            ax.annotate(v.read_bytes, [k, v.read_bytes])
        else:
            ax.annotate(v.write_bytes, [k, v.write_bytes])

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AXI Performance Monitor Convolutions Data visualization')
    parser.add_argument('--log_file', type=str, default="logs/log.json",
                        help='input log file path')
    parser.add_argument('--workloads', type=str, default="0,1",
                        help='workloads to compare and plot')
    parser.add_argument('--boxplot', action='store_true',
                        help='generate box plots of read b/w of workloads')
    parser.add_argument('--plot_write_bytes', action='store_true',
                        help='generate plot of write bytes of workloads')
    parser.add_argument('--plot_read_bytes', action='store_true',
                        help='generate plot of read bytes of workloads')
    parser.add_argument('--plot_read_bytes_uop', action='store_true',
                        help='generate plot of read bytes from compute uop of workloads')
    parser.add_argument('--plot_read_bytes_data', action='store_true',
                        help='generate plot of read bytes from compute data of workloads')
    args = parser.parse_args()

    log_file = args.log_file
    workloads = [int(workload) for workload in args.workloads.split(',')]

    data = []
    with open(log_file, 'r') as myfile:
        file_data = json.load(myfile)
        for wkl_idx in workloads:
            data.append(file_data["workloads"][wkl_idx])

    if args.boxplot:
        generate_boxplot(data)

    if args.plot_write_bytes:
        plot_bytes(data, '5', 'write')

    if args.plot_read_bytes:
        plot_bytes(data, '2', 'read')

    if args.plot_read_bytes_uop:
        plot_bytes(data,'3','read')

    if args.plot_read_bytes_data:
        plot_bytes(data,'4','read')

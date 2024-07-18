import pandas as pd
import matplotlib.pyplot as plt

# Read the data from a CSV file
data = pd.read_csv('fft_performance_report.csv')

# Function to plot elapsed time vs signal size for different precisions, grouped by batch size
def plot_elapsed_time_vs_signal_size(data):
    batch_sizes = data['Batch Size'].unique()
    precisions = data['Precision'].unique()
    signal_sizes = sorted(data['Signal Size'].unique())
    
    fig, axs = plt.subplots(len(batch_sizes), 3, figsize=(15, 5 * len(batch_sizes)))
    for i, batch_size in enumerate(batch_sizes):
        subset = data[data['Batch Size'] == batch_size]
        for j, task in enumerate(['Creation_FFT', 'Creation_IFFT', 'FFT-Iteration 1', 'IFFT-Iteration 1']):
            ax = axs[i, j % 3] if len(batch_sizes) > 1 else axs[j % 3]
            for precision in precisions:
                task_data = subset[(subset['Precision'] == precision) & (subset['Task'] == task)]
                ax.plot(task_data['Signal Size'], task_data['Elapsed Time (ms)'], label=precision)
            ax.set_title(f'Batch Size: {batch_size}, Task: {task}')
            ax.set_xlabel('Signal Size')
            ax.set_ylabel('Elapsed Time (ms)')
            ax.set_xticks(signal_sizes)
            ax.grid(True, which='both', axis='x', linestyle=':', color='gray')
            ax.legend()
    plt.tight_layout()
    plt.savefig('fft_performance_report.png' , dpi=300)

# Function to plot elapsed time vs batch size for different precisions, grouped by signal size
def plot_elapsed_time_vs_batch_size(data):
    batch_sizes = sorted(data['Batch Size'].unique())
    precisions = data['Precision'].unique()
    signal_sizes = data['Signal Size'].unique()
    
    fig, axs = plt.subplots(len(signal_sizes), 3, figsize=(15, 5 * len(signal_sizes)))
    for i, signal_size in enumerate(signal_sizes):
        subset = data[data['Signal Size'] == signal_size]
        for j, task in enumerate(['Creation_FFT', 'Creation_IFFT', 'FFT-Iteration 1', 'IFFT-Iteration 1']):
            ax = axs[i, j % 3] if len(signal_sizes) > 1 else axs[j % 3]
            for precision in precisions:
                task_data = subset[(subset['Precision'] == precision) & (subset['Task'] == task)]
                ax.plot(task_data['Batch Size'], task_data['Elapsed Time (ms)'], label=precision)
            ax.set_title(f'Signal Size: {signal_size}, Task: {task}')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Elapsed Time (ms)')
            ax.set_xticks(batch_sizes)
            ax.grid(True, which='both', axis='x', linestyle=':', color='gray')
            ax.legend()
    plt.tight_layout()
    plt.savefig('fft_performance_report.png' , dpi=300)

# Call the plotting functions
plot_elapsed_time_vs_signal_size(data)
plot_elapsed_time_vs_batch_size(data)

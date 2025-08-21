import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import os

def plot_training_time(means, stds, labels, fig_name, title="Training Time"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.7, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Execution Time (Second)')
    ax.set_title(title)
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

def plot_tokens_per_second(means, stds, labels, fig_name, title="Tokens Per Second"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.7, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Throughput (Tokens per Second)')
    ax.set_title(title)
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# Fill the data points here
if __name__ == '__main__':

    os.makedirs('../submit_figures', exist_ok=True)
    
    # Problem 1.3: Data Parallel results
    dp_rank0_time_mean, dp_rank0_time_std = 54.77310221195221, 0.3471499919070063
    dp_rank1_time_mean, dp_rank1_time_std = 54.011241507530215, 0.11739255129184738
    single_gpu_time_mean, single_gpu_time_std = 64.60698866844177, 0.08927946904521586
    dp_avg_time_mean = (dp_rank0_time_mean + dp_rank1_time_mean) / 2
    dp_avg_time_std = np.sqrt((dp_rank0_time_std**2 + dp_rank1_time_std**2) / 2)

    dp_rank0_tokens_mean, dp_rank0_tokens_std = 42132.48983044551, 2924.1588824778023
    dp_rank1_tokens_mean, dp_rank1_tokens_std = 40768.08564737876, 1138.485103635409
    single_gpu_tokens_mean, single_gpu_tokens_std = 44332.581035688985, 3688.6059507999203
    dp_total_tokens_mean = dp_rank0_tokens_mean + dp_rank1_tokens_mean
    dp_total_tokens_std = np.sqrt(dp_rank0_tokens_std**2 + dp_rank1_tokens_std**2)
    
    plot_training_time([dp_avg_time_mean, single_gpu_time_mean],
                      [dp_avg_time_std, single_gpu_time_std],
                      ['Data Parallel - 2GPUs', 'Single GPU'],
                      '../submit_figures/ddp_vs_single_training_time.png',
                      'Training Time')
    plot_tokens_per_second([dp_total_tokens_mean, single_gpu_tokens_mean],
                          [dp_total_tokens_std, single_gpu_tokens_std],
                          ['Data Parallel - 2GPUs', 'Single GPU'],
                          '../submit_figures/ddp_vs_single_tokens_per_second.png',
                          'Tokens Per Second')
    
    # Problem 2.3: Pipeline Parallel results
    pp_time_mean, pp_time_std = 49.87996590137482, 0.759651780128479
    pp_tokens_mean, pp_tokens_std = 12833.779351488196, 195.45328778713974
    
    mp_time_mean, mp_time_std = 65.7433660030365, 0.2332477569580078
    mp_tokens_mean, mp_tokens_std = 9734.945057665107, 34.53814790537126
    
    plot_training_time([pp_time_mean, mp_time_mean],
                      [pp_time_std, mp_time_std],
                      ['Pipeline Parallel', 'Model Parallel'],
                      '../submit_figures/pp_vs_mp_training_time.png',
                      'Training Time')
    plot_tokens_per_second([pp_tokens_mean, mp_tokens_mean],
                          [pp_tokens_std, mp_tokens_std],
                          ['Pipeline Parallel', 'Model Parallel'],
                          '../submit_figures/pp_vs_mp_tokens_per_second.png',
                          'Tokens Per Second')
    
    print("All plots have been generated and saved to submit_figures directory!")

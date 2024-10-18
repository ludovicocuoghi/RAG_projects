import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from filter_chunks_reranked import filter_chunks_reranked

def plot_comparisons(reranked_results, retrieved_results):
    # Extract information from reranked_results
    reranked_sims = reranked_sims = [np.round(next((res['score'] for res in retrieved_results if res['id'] == result['id']), None), 2) for result in reranked_results]
    reranked_scores = [np.round(result["score"], 2) for result in reranked_results]
    retrieved_sims = [np.round(result["score"], 2) for result in retrieved_results]
    positions = np.arange(1, len(retrieved_results) + 1) 

    # Determine selected indices using the existing filtering logic
    selected_indices = filter_chunks_reranked(reranked_results)
    last_selected_index = len(selected_indices)  # Position of the last selected index

    sns.set(style="whitegrid")
    plt.rcParams.update({
        'axes.labelsize': 8,  
        'axes.titlesize': 10,
        'xtick.labelsize': 8,  
        'ytick.labelsize': 8,  
        'legend.fontsize': 8,  
        'lines.linewidth': 2.5,  
        'lines.markersize': 6 
    })

    # Plotting
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 5.5))

    sns.lineplot(x=positions, y=retrieved_sims, ax=axs[0], marker='o', linestyle='-', color='skyblue')
    sns.lineplot(x=positions, y=reranked_sims, ax=axs[0], label='After Reranking', marker='s', linestyle='--', color='darkorange')
    axs[0].set_title('Comparison of Cosine Similarities Before and After Reranking')
    axs[0].set_xlabel('Chunk ID (Reordered by Cosine Similarity)')
    axs[0].set_ylabel('Cosine Similarity')
    axs[0].set_xticks(np.arange(1, len(positions) + 1))
    axs[0].legend(loc='upper right')  
    axs[0].set_ylim(0, 1)
    axs[0].grid(True, which='major', linestyle='--', linewidth=0.5) 

    #Adding comparison of top chunks
    top_k = [5, 10, 15]
    comparison_text = ""
    for k in top_k:
        if k <= len(retrieved_results) and k <= len(reranked_results):
            retrieved_ids = [result['id'] for result in retrieved_results[:k]]
            reranked_ids = [result['id'] for result in reranked_results[:k]]
            common_ids = set(retrieved_ids).intersection(set(reranked_ids))
            comparison_text += f"Common IDs in top {k}: {len(common_ids)}/{k}\n"

    axs[0].text(0.95, 0.05, comparison_text.strip(), transform=axs[0].transAxes, fontsize=10, verticalalignment='bottom', 
                horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # Reranked Scores with Threshold Line
    sns.lineplot(x=positions, y=reranked_scores, ax=axs[1], label='Reranked Score', color='limegreen', marker='o')
    axs[1].axvline(x=last_selected_index, color='indianred', linestyle='--', linewidth=2, label='Score Threshold')
    axs[1].text(last_selected_index + 0.2, np.max(reranked_scores) * 0.85, 'THRESHOLD', horizontalalignment='left', color='indianred', fontsize=8, fontweight='bold')
    axs[1].set_title('Reranked Scores with Threshold Indicator')
    axs[1].set_xlabel('Chunk ID (Reordered by Reranked Score)')
    axs[1].set_ylabel('Score')
    axs[1].set_ylim(0, 1.2)  
    axs[1].set_yticks(np.arange(0, 1.1, 0.2))  
    axs[1].set_xticks(np.arange(1, len(positions) + 1))
    axs[1].legend(loc='upper right')  
    axs[1].grid(True, which='major', linestyle='--', linewidth=0.5) 

    plt.tight_layout()
    st.pyplot(fig)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from filter_chunks_reranked import filter_chunks_reranked

def plot_comparisons(reranked_results, retrieved_sims):
    # Extract information from reranked_results
    reranked_sims = [np.round(result["cosine_similarity"], 2) for result in reranked_results]
    reranked_scores = [np.round(result["score"], 2) for result in reranked_results]
    positions = np.arange(1, len(retrieved_sims) + 1)  # Define positions for x-axis

    # Determine selected indices using the existing filtering logic
    selected_indices = filter_chunks_reranked(reranked_results)
    last_selected_index = len(selected_indices)  # Position of the last selected index

    # Plotting
    fig, axs = plt.subplots(ncols=2, figsize=(16, 6))

    # Plot 1: Cosine Similarities
    sns.lineplot(x=positions, y=retrieved_sims, ax=axs[0], label='Original Similarities', marker='o')
    sns.lineplot(x=positions, y=reranked_sims, ax=axs[0], label='Reranked Similarities', marker='o')
    axs[0].set_title('Comparison of Cosine Similarities: Original vs Reranked')
    axs[0].set_xlabel('Position')
    axs[0].set_ylabel('Cosine Similarity')
    axs[0].set_xticks(positions)  # Set x-ticks to positions
    axs[0].legend()

    # Plot 2: Reranked Scores with a single vertical line for the last selected index
    sns.lineplot(x=positions, y=reranked_scores, ax=axs[1], label='Reranked Scores', color='green', marker='o')
    axs[1].axvline(x=last_selected_index, color='red', linestyle='--', linewidth=2, label='Filter Cutoff')
    axs[1].set_title('Reranked Scores with Filter Cutoff')
    axs[1].set_xlabel('Position')
    axs[1].set_ylabel('Scores')
    axs[1].set_xticks(positions)  # Set x-ticks to positions
    axs[1].legend()

    plt.tight_layout()
    st.pyplot(fig)

# Example usage in a Streamlit script
def main():
    # Example input data
    original_sims = [0.79, 0.76, 0.72, 0.72, 0.71, 0.7, 0.7, 0.7, 0.68, 0.68, 0.68, 0.67, 0.63, 0.62, 0.62, 0.58, 0.57, 0.54, 0.54, 0.53, 0.52, 0.51, 0.51, 0.51, 0.51]
    reranked_results = [{'id': '13', 'sim': 0.79, 'score': 1.0},
                        {'id': '16', 'sim': 0.76, 'score': 0.98},
                        # Continue with other reranked results
                       ]
    
    st.title('Visualization of Cosine Similarities, Reranked Scores, and Index Position Shifts')
    plot_comparisons(reranked_results, original_sims)

if __name__ == "__main__":
    main()

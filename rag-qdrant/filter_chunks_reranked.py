import numpy as np


DROP_THERSHOLD=0.5
PERCENTILE_THERSHOLD=0.9
MIN_SCORE_THERSHOLD=0.1

def filter_chunks_reranked(reranked_results,
                            drop_threshold=DROP_THERSHOLD,
                            percentile_threshold=PERCENTILE_THERSHOLD,
                            min_score_threshold=MIN_SCORE_THERSHOLD):
    # Extract reranked scores
    reranked_scores = [result["score"] for result in reranked_results]
    
    # Early exit if the maximum score is below the minimum threshold
    if max(reranked_scores) < min_score_threshold:
        return []  # or return [reranked_results[0]] if you want to always include at least one
    
    # Step 1: Identify significant drop with a stricter threshold
    drop_filtered_indices = []
    for i in range(len(reranked_scores) - 1):
        drop_filtered_indices.append(i)
        if reranked_scores[i] - reranked_scores[i + 1] > drop_threshold:
            break
    
    # Ensure the first chunk is always included
    if 0 not in drop_filtered_indices:
        drop_filtered_indices.insert(0, 0)
    
    # Step 2: Apply percentile-based filtering with a higher threshold
    score_threshold = np.percentile(reranked_scores, percentile_threshold * 100)
    percentile_filtered_indices = [i for i, score in enumerate(reranked_scores) if score >= score_threshold]
    
    # Combine both methods
    combined_indices = sorted(set(drop_filtered_indices).union(percentile_filtered_indices))
    
    # Step 3: Filter out chunks with scores below the minimum acceptable score
    final_selection = [reranked_results[i] for i in combined_indices if reranked_results[i]["score"] >= min_score_threshold]

    return final_selection

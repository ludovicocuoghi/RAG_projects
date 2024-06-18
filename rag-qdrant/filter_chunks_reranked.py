def filter_chunks_reranked(reranked_results,
                           high_score_threshold=0.8,
                           soft_score_threshold=0.4,
                           low_score_threshold=0.2,
                           drop_threshold=0.4,
                           min_chunks=5):
    # Reorder reranked results by score in descending order
    reranked_results = sorted(reranked_results, key=lambda x: x["score"], reverse=True)
    reranked_scores = [result["score"] for result in reranked_results]

    selected_indices = []
    prev_score = None

    for i, score in enumerate(reranked_scores):
        if score >= high_score_threshold:
            selected_indices.append(i)
        elif score >= soft_score_threshold:
            if prev_score is not None and (prev_score - score) > drop_threshold:
                break  # Stop if drop threshold exceeded
            selected_indices.append(i)
        elif score < low_score_threshold:
            break  # Stop adding if score is below low score threshold
        prev_score = score

    # Ensure at least min_chunks are selected, provided they meet the low score threshold
    if len(selected_indices) < min_chunks:
        additional_indices = [i for i in range(len(reranked_results))
                              if i not in selected_indices and reranked_results[i]["score"] >= low_score_threshold]
        selected_indices += additional_indices[:max(0, min_chunks - len(selected_indices))]

    final_selection = [reranked_results[i] for i in selected_indices]
    return final_selection

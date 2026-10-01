class Metrics:
    def recall_at_k(self, found_doc_ids, relevant_docs, k):
        if not relevant_docs:
            return 0.0
        retrieved_at_k = set(found_doc_ids[:k])
        relevant_retrieved = set(relevant_docs)
        intersection = retrieved_at_k.intersection(relevant_retrieved)
        return len(intersection) / len(relevant_docs)

    def reciprocal_rank(self, found_doc_ids, relevant_docs):
        relevant = set(relevant_docs)
        for rank, doc_id in enumerate(found_doc_ids, start=1):
            if doc_id in relevant:
                return 1 / rank
        return 0.0
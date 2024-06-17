from typing import List
from rank_bm25 import BM25Okapi


class Retriever():
    def __init__(self, corpus_cov: List[str], corpus_fm: List[str], corpus_tc: List[str]) -> None:
        super().__init__()
        self.corpus_cov = corpus_cov
        self.corpus_fm = corpus_fm
        self.corpus_tc = corpus_tc
        self.corpus_fm_base = [doc.split() for doc in corpus_fm]
        self.corpus_tc_base = [doc.split() for doc in corpus_tc]
        self.corpus_cov_base = [doc.split() for doc in corpus_cov]
        self.bm25_fm = BM25Okapi(self.corpus_fm_base)
        self.bm25_tc = BM25Okapi(self.corpus_tc_base)
        self.bm25_cov = BM25Okapi(self.corpus_cov_base)

    def retrieve(self, target_fm: str, top_k: int = 3, mode: str = 'fm'):
        assert mode in ['fm', 'tc', 'both']
        if mode == 'fm':
            scores = self.bm25_fm.get_scores(target_fm.split())
            top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        elif mode == 'tc':
            scores = self.bm25_tc.get_scores(target_fm.split())
            top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        else:
            scores_fm = self.bm25_fm.get_scores(target_fm.split())
            scores_tc = self.bm25_tc.get_scores(target_fm.split())
            scores = [0.5 * scores_fm[i] + 0.5 * scores_tc[i] for i in range(len(scores_fm))]
            top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [self.corpus_cov[i] for i in top_k_indices], [self.corpus_fm[i] for i in top_k_indices], [self.corpus_tc[i] for i in top_k_indices], [scores[i] for i in top_k_indices]
    

if __name__ == "__main__":
    import json
    with open('./data/raw_data/spark_valid_pairs.json', 'r') as f:
        valid_pairs = json.load(f)

    corpus_fm, corpus_tc = [], []
    for each_pair in valid_pairs:
        each_file_path, method_name, test_case, focal_method = each_pair[0], each_pair[1], each_pair[2], each_pair[3]
        corpus_fm.append(focal_method)
        corpus_tc.append(test_case)
    
    retriever = Retriever(corpus_fm, corpus_tc)
    target_fm = "public static String decode(String toDecodeContent) {\n         if (toDecodeContent == null) {\n             return null;\n         }\n         byte[] buf = null;\n         try {\n             buf = decoder.decode(toDecodeContent);\n         } catch (Exception e) {\n             e.printStackTrace();\n         }\n         return new String(buf);\n     }"
    ref_fms, ref_tcs = retriever.retrieve(target_fm)
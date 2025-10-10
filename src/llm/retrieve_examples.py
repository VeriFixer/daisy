import utils.global_variables as gl
from llm.extract_error_blocks import extract_error_blocks
import pickle
import json
from tqdm import tqdm
from pathlib import Path
 
# ── Generation ─────────────────────────────────────────────────────────────────
def generate_and_pickle(dataset_dir: Path, model):
    from sentence_transformers import SentenceTransformer
    import torch
    from sentence_transformers.util import cos_sim
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    prog_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    rows = []
    corpus = []
    for prog_dir in tqdm(prog_dirs, desc="Creating Full dataset Embeddings", total=len(prog_dirs)):
        for grp_dir in prog_dir.iterdir():
            if not grp_dir.is_dir() or grp_dir.name in ("bin", "obj"):
                continue

            error_txt = (grp_dir / "verifier_output.txt").read_text(encoding='utf-8')
            error_txt_filter = extract_error_blocks( error_txt)

            code_txt  = (grp_dir / "method_with_assertion_placeholder.dfy").read_text(encoding='utf-8')
            corpus.append(code_txt)
            
            assertions = str(json.load(open(grp_dir / "oracle_assertions.json")))

            oracle_pos = (grp_dir / "oracle_fix_position.txt").read_text(encoding='utf-8')
            # embed on GPU, then move back to CPU for pickling
            err_emb = model.encode([error_txt_filter], convert_to_tensor=True).cpu()
            cod_emb = model.encode([code_txt],  convert_to_tensor=True).cpu()

            # pickle per-group
            with open(grp_dir / "error_embeds.pkl", "wb") as f:
                pickle.dump(err_emb, f)
            with open(grp_dir / "code_embeds.pkl", "wb") as f:
                pickle.dump(cod_emb, f)

            rows.append({
                "prog": prog_dir.name,
                "group": grp_dir.name,
                "error_message":  error_txt_filter,
                "code_snippet": code_txt,
                "assertions": assertions,
                "oracle_pos" : oracle_pos,
                # keep the tensors for immediate use if you want
                "error_embeds": err_emb,
                "code_embeds": cod_emb
            })

     # After gathering all entries, build TF-IDF
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r"[\w@]+", analyzer="word")
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Save vectorizer and matrix once
    with open(dataset_dir / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(dataset_dir / "tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)

    return rows, tfidf_vectorizer, tfidf_matrix 

# ── Loading ────────────────────────────────────────────────────────────────────
def load_entries_from_pickles(dataset_dir: Path):
    entries = []
    prog_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    for prog_dir in prog_dirs:
        for grp_dir in prog_dir.iterdir():
            if not grp_dir.is_dir() or grp_dir.name in ("bin", "obj"):
                continue

            # load the pickled embeddings (which were CPU tensors)
            with open(grp_dir / "error_embeds.pkl", "rb") as f:
                 err_emb = pickle.load(f)
            with open(grp_dir / "code_embeds.pkl", "rb") as f:
                cod_emb = pickle.load(f)

            entries.append({
                "prog": prog_dir.name,
                "group": grp_dir.name,
                "error_message": (grp_dir / "verifier_output.txt").read_text(encoding='utf-8'),
                "code_snippet":  (grp_dir / "method_with_assertion_placeholder.dfy").read_text(encoding='utf-8'),
                "oracle_pos" : (grp_dir / "oracle_fix_position.txt").read_text(encoding='utf-8'),
                "assertions": str(json.load(open(grp_dir / "oracle_assertions.json"))),
                "error_embeds": err_emb,
                "code_embeds": cod_emb
            })
    with open(dataset_dir / "tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    with open(dataset_dir / "tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    return entries, tfidf_vectorizer, tfidf_matrix 

# ── Retrieval ──────────────────────────────────────────────────────────────────

def normalize_minmax(sim):
    from sentence_transformers import SentenceTransformer
    import torch
    from sentence_transformers.util import cos_sim
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    minv, maxv = sim.min(), sim.max()
    if maxv - minv > 0:
        return (sim - minv) / (maxv - minv)
    else:
        return torch.zeros_like(sim)
    
def retrieve_by_error_and_code(
    new_error: str,
    new_code: str,
    entries: list[dict],
    top_k: int = 5,
    method: str = "error_code",  # "error_code", "embedding", or "tfidf"
    α: float = 0.5,
    prog_original = None,
    group_original = None,
    model=None,
    device=None,
    diferent_methods=1,
    tfidf_vectorizer= None,
    tfidf_matrix= None
):
    from sentence_transformers import SentenceTransformer
    import torch
    from sentence_transformers.util import cos_sim
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if(top_k == -1):
        top_k = len(entries)

    if method == "tfidf":
        if tfidf_vectorizer is None or tfidf_matrix is None:
            raise ValueError("tfidf_vectorizer and tfidf_matrix must be provided for tfidf method")

        q_vec = tfidf_vectorizer.transform([new_code])
        sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
        scores = torch.tensor(sims)
        idxs = torch.topk(scores, k=top_k).indices.tolist()
        selected_scores = scores[idxs].tolist()
    else:
        q_cod = model.encode([new_code], convert_to_tensor=True).to(device)    
        all_codes = torch.cat([e["code_embeds"]  for e in entries], dim=0).to(device)
        sim_cod = cos_sim(q_cod, all_codes)[0]
        sim_cod_n = normalize_minmax(sim_cod)

        if(method == "embedding"):
            combined = sim_cod_n      
        else:
            q_err = model.encode([new_error], convert_to_tensor=True).to(device)
            all_errs  = torch.cat([e["error_embeds"] for e in entries], dim=0).to(device)
            sim_err = cos_sim(q_err,  all_errs)[0]
            sim_err_n = normalize_minmax(sim_err)
            combined = α * sim_err_n + (1 - α) * sim_cod_n    
        idxs = torch.topk(combined, k=top_k).indices.tolist()
        selected_scores = combined[idxs].tolist()


    if prog_original and group_original:
        _, _, orig_method_start ,*_ = group_original.split("_", 3)
    else:
         orig_method_start= None

    results = []
    inserted_progs = []
    for i, score in zip(idxs, selected_scores):
        e = entries[i]
        if prog_original and group_original:
            _, _, method_start ,*_ = e["group"].split("_", 3)
            if(diferent_methods == 1 and (e["prog"], method_start) in inserted_progs):
                continue # method already inserted in other examples
            if e["prog"] == prog_original:
                _, _, method_startstart ,*_ = e["group"].split("_", 3)
                if method_start == orig_method_start: # They represent the sam emethod can be skipped
                    continue
                
            inserted_progs.append((e["prog"], method_start))
        
        results.append({
            "prog": e["prog"],
            "group": e["group"],
            "score": score,
            "error_message": e["error_message"],
            "code_snippet": e["code_snippet"],
            "assertions" : e["assertions"],
            "oracle_pos" : e["oracle_pos"]
             
        })
    return results

def generate_example_model():
    from sentence_transformers import SentenceTransformer
    import torch
    from sentence_transformers.util import cos_sim
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    # Use a static-like attribute on the function itself (first time is called is created rest it gives away already created variable)
    if not hasattr(generate_example_model, "_entries"):
        # ── Config ────────────────────────────────────────────────────────────────
        GENERATE_DATASET_EMBEDDINGS = gl.GENERATE_DATASET_EMBEDDINGS
        DATASET_DIR = Path(gl.DAFNY_ASSERTION_DATASET)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Model ─────────────────────────────────────────────────────────────────
        model = SentenceTransformer(
            'jinaai/jina-embeddings-v2-base-code',
            trust_remote_code=True,
            device=device
        )
        generate_example_model._model = model
        generate_example_model._device = device 
        if GENERATE_DATASET_EMBEDDINGS:
            generate_example_model._entries, generate_example_model._tfidf_vectorizer, generate_example_model._tfidf_matrix = generate_and_pickle(gl.DAFNY_ASSERTION_DATASET, model)
        else:
            # Only generates once if GENRATE is not active
            try:
                 generate_example_model._entries, generate_example_model._tfidf_vectorizer, generate_example_model._tfidf_matrix  = load_entries_from_pickles(gl.DAFNY_ASSERTION_DATASET)
            except:
                 generate_example_model._entries, generate_example_model._tfidf_vectorizer, generate_example_model._tfidf_matrix = generate_and_pickle(gl.DAFNY_ASSERTION_DATASET, model)       

    return generate_example_model._entries,  generate_example_model._model,  generate_example_model._device,  generate_example_model._tfidf_vectorizer, generate_example_model._tfidf_matrix 

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Demo query ─────────────────────────────────────────────────────────────
    test_error_filter = """
    /tmp/dafny_thread_2045894_zbrybjfa/temp_2045894_138543117502336.dfy(11,20): Error: this invariant could not be proved to be maintained by the loop
 Related message: loop invariant violation
   |
11 |     invariant count == |set i | i in grow && i < threshold|
   |               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Dafny program verifier finished with 0 verified, 1 error
"""

    test_code  = """method CountLessThan(numbers: set<int>, threshold: int) returns (count: int)
  ensures count == |set i | i in numbers && i < threshold|
{
  count := 0;
  var shrink := numbers;
  var grow := {};
  while |shrink | > 0
    decreases shrink
    invariant shrink + grow == numbers
    invariant grow !! shrink
    invariant count == |set i | i in grow && i < threshold|
  {
    var i: int :| i in shrink;
    shrink := shrink - {i};
    var grow' := grow+{i};
    /*<Assertion is Missing Here>*/
    grow := grow + {i};
    if i < threshold {
      count := count + 1;
    }
  }
}"""

    # Load corpus and models
    entries, model, device, tfidf_vectorizer, tfidf_matrix = generate_example_model()

    prog_orig = "Clover_count_lessthan_dfy"
    group_orig = "method_start_0_as_start_460_end_591"

    prog_orig = None
    group_orig = None
    def print_top(results, method_desc):
        print(f"\n=== {method_desc} ===")
        for rank, r in enumerate(results[:3], 1):
            print(f"{rank}. [{r['score']:.3f}] PROG={r['prog']} GROUP={r['group']}")
            print(f"   Error:\n{r['error_message']}")
            print(f"   Code:\n{r['code_snippet']}\n")
        print("------------------------------------------------------------------------")

    # 1. Embedding only
    results_emb = retrieve_by_error_and_code(
        test_error_filter, test_code, entries,
        method="embedding", top_k=-1,
        prog_original=prog_orig, group_original=group_orig,
        model=model, device=device
    )
    print_top(results_emb, "Embedding only")

    # 2. Error+Code hybrid (α ∈ {0.0, 0.25, 0.5, 0.75, 1.0})
    # No need 0 that 0 is equal to embedding plain
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        results_hybrid = retrieve_by_error_and_code(
            test_error_filter, test_code, entries,
            method="error_code", α=alpha, top_k=-1,
            prog_original=prog_orig, group_original=group_orig,
            model=model, device=device
        )
        print_top(results_hybrid, f"Hybrid error_code (α={alpha:.2f})")

    # 3. TF-IDF
    results_tfidf = retrieve_by_error_and_code(
        test_error_filter, test_code, entries,
        method="tfidf", top_k=-1,
        prog_original=prog_orig, group_original=group_orig,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix
    )
    print_top(results_tfidf, "TF-IDF only")

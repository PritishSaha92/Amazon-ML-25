import os
import sys
import argparse
import json
import gc
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy import sparse

# Sklearn fallbacks (CPU)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD as SKTruncatedSVD
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
import joblib


def _read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def _normalize_text_series(texts: pd.Series) -> pd.Series:
    # Lightweight normalization tailored for TF-IDF; preserve case-insensitive signals via lowercase=True in vectorizers
    texts = texts.fillna("").astype(str)
    # Collapse whitespace to reduce char-ngram noise
    texts = texts.str.replace(r"\s+", " ", regex=True).str.strip()
    return texts


def build_tfidf_matrices(
    train_text: pd.Series,
    test_text: pd.Series,
    word_ngram_range: Tuple[int, int] = (1, 2),
    char_ngram_range: Tuple[int, int] = (3, 5),
    max_features_word: int = 200_000,
    max_features_char: int = 300_000,
    min_df_word: int = 3,
    max_df_word: float = 0.98,
    min_df_char: int = 3,
    max_df_char: float = 1.0,
    strip_accents: Optional[str] = "unicode",
    lowercase: bool = True,
    dtype=np.float32,
    artifacts_dir: Optional[str] = None,
    random_state: int = 42,
):
    """Fit TF-IDF on train only; transform test; return stacked CSR matrices and fitted vectorizers.

    Returns: (Xtr, Xte, word_vec, char_vec)
    """
    train_text = _normalize_text_series(train_text)
    test_text = _normalize_text_series(test_text)

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram_range,
        max_features=max_features_word,
        min_df=min_df_word,
        max_df=max_df_word,
        lowercase=lowercase,
        strip_accents=strip_accents,
        sublinear_tf=True,
        dtype=dtype,
        norm="l2",
    )
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=char_ngram_range,
        max_features=max_features_char,
        min_df=min_df_char,
        max_df=max_df_char,
        lowercase=lowercase,
        strip_accents=strip_accents,
        sublinear_tf=True,
        dtype=dtype,
        norm="l2",
    )

    print(f"Fitting word TF-IDF {word_ngram_range}, max_features={max_features_word}, min_df={min_df_word}, max_df={max_df_word} ...")
    Xw_tr = word_vec.fit_transform(train_text)
    Xw_te = word_vec.transform(test_text)
    print(f"  word tfidf: train={Xw_tr.shape}, test={Xw_te.shape}, nnz_train={Xw_tr.nnz}")

    print(f"Fitting char TF-IDF {char_ngram_range}, max_features={max_features_char}, min_df={min_df_char} ...")
    Xc_tr = char_vec.fit_transform(train_text)
    Xc_te = char_vec.transform(test_text)
    print(f"  char tfidf: train={Xc_tr.shape}, test={Xc_te.shape}, nnz_train={Xc_tr.nnz}")

    Xtr = sparse.hstack([Xw_tr, Xc_tr], format="csr", dtype=dtype)
    Xte = sparse.hstack([Xw_te, Xc_te], format="csr", dtype=dtype)
    print(f"Stacked TF-IDF: train={Xtr.shape}, test={Xte.shape}, nnz_train={Xtr.nnz}")

    if artifacts_dir:
        os.makedirs(artifacts_dir, exist_ok=True)
        joblib.dump(word_vec, os.path.join(artifacts_dir, "tfidf_word_vectorizer.joblib"))
        joblib.dump(char_vec, os.path.join(artifacts_dir, "tfidf_char_vectorizer.joblib"))
        # Save vocab sizes for quick reference
        meta = {
            "word_vocab_size": len(word_vec.vocabulary_),
            "char_vocab_size": len(char_vec.vocabulary_),
            "word_ngram_range": list(word_ngram_range),
            "char_ngram_range": list(char_ngram_range),
        }
        with open(os.path.join(artifacts_dir, "tfidf_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # Free intermediates we no longer need
    del Xw_tr, Xw_te, Xc_tr, Xc_te
    gc.collect()

    return Xtr, Xte, word_vec, char_vec


def _try_gpu_sparse_svd(
    Xtr_csr: sparse.csr_matrix,
    Xte_csr: sparse.csr_matrix,
    n_components: int,
    random_state: int,
    n_iter: int,
):
    """Attempt a GPU sparse SVD using cuML randomized SVD (CSR) or a CuPy implementation.

    Returns dense float32 arrays if successful; otherwise None.
    """
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpx_sparse
        # Try cuML randomized SVD first (two import paths)
        cuml_rand_svd = None
        try:
            from cuml.sparse.linalg import randomized_svd as _cuml_rand_svd
            cuml_rand_svd = _cuml_rand_svd
        except Exception:
            try:
                from cuml.experimental.sparse.linalg import randomized_svd as _cuml_rand_svd  # type: ignore
                cuml_rand_svd = _cuml_rand_svd
            except Exception:
                cuml_rand_svd = None

        # Transfer sparse matrices to GPU
        Xtr_gpu = cpx_sparse.csr_matrix(
            (cp.asarray(Xtr_csr.data), cp.asarray(Xtr_csr.indices), cp.asarray(Xtr_csr.indptr)),
            shape=Xtr_csr.shape,
        )
        Xte_gpu = cpx_sparse.csr_matrix(
            (cp.asarray(Xte_csr.data), cp.asarray(Xte_csr.indices), cp.asarray(Xte_csr.indptr)),
            shape=Xte_csr.shape,
        )

        if cuml_rand_svd is not None:
            print("Using GPU cuML sparse randomized SVD...")
            U_gpu, S_gpu, Vt_gpu = cuml_rand_svd(
                Xtr_gpu,
                n_components=n_components,
                n_iter=n_iter,
                random_state=random_state,
            )
            Vk_gpu = Vt_gpu.T  # (d, k)
            Xtr_red_gpu = Xtr_gpu.dot(Vk_gpu)
            Xte_red_gpu = Xte_gpu.dot(Vk_gpu)
            Xtr_red = cp.asnumpy(Xtr_red_gpu).astype(np.float32, copy=False)
            Xte_red = cp.asnumpy(Xte_red_gpu).astype(np.float32, copy=False)
            return Xtr_red, Xte_red

        # Fallback: pure CuPy randomized SVD (Halko) on sparse CSR
        print("cuML sparse randomized SVD not found; using CuPy randomized SVD...")
        cp.random.seed(random_state)
        k = int(n_components)
        p = max(32, min(128, k // 8))  # oversampling
        b = k + p

        # Draw random gaussian matrix for sketching
        Omega = cp.random.standard_normal((Xtr_gpu.shape[1], b), dtype=cp.float32)
        # Y = A * Omega
        Y = Xtr_gpu.dot(Omega)  # (n, b)
        # Orthonormalize
        Q, _ = cp.linalg.qr(Y, mode="reduced")
        for _ in range(max(0, n_iter)):
            Z = Xtr_gpu.T.dot(Q)  # (d, b)
            Q, _ = cp.linalg.qr(Xtr_gpu.dot(Z), mode="reduced")

        # Small dense SVD on B = Q^T A  (b x d). Ensure dense CuPy array.
        B = (Q.T @ Xtr_gpu).toarray()  # (b, d) cupy.ndarray
        # Compute top-k via SVD on small side (b x d)
        Ub, S, Vh = cp.linalg.svd(B, full_matrices=False)  # Ub:(b,b) S:(b,) Vh:(b,d)
        Vk = Vh[:k, :].T  # (d, k)

        # Project train/test: X * V
        Xtr_red_gpu = Xtr_gpu.dot(Vk)
        Xte_red_gpu = Xte_gpu.dot(Vk)

        Xtr_red = cp.asnumpy(Xtr_red_gpu).astype(np.float32, copy=False)
        Xte_red = cp.asnumpy(Xte_red_gpu).astype(np.float32, copy=False)
        return Xtr_red, Xte_red
    except Exception as e:
        print(f"GPU sparse SVD not available or failed ({e}); falling back to CPU.")
        return None


def reduce_with_svd(
    Xtr: sparse.csr_matrix,
    Xte: sparse.csr_matrix,
    n_components: int,
    random_state: int = 42,
    n_iter: int = 7,
    center_and_normalize: bool = True,
    artifacts_dir: Optional[str] = None,
):
    """Reduce sparse TF-IDF with TruncatedSVD (train-only fit) and optional scaling/normalization."""
    assert n_components > 0, "n_components must be positive when using SVD"
    # Try GPU path first (sparse randomized SVD)
    gpu_res = _try_gpu_sparse_svd(
        Xtr, Xte, n_components=n_components, random_state=random_state, n_iter=n_iter
    )
    if gpu_res is not None:
        Xtr_red, Xte_red = gpu_res
    else:
        print("Fitting CPU TruncatedSVD...")
        tsvd = SKTruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=random_state)
        Xtr_red = tsvd.fit_transform(Xtr)
        Xte_red = tsvd.transform(Xte)
        if artifacts_dir:
            os.makedirs(artifacts_dir, exist_ok=True)
            joblib.dump(tsvd, os.path.join(artifacts_dir, f"tfidf_svd_{n_components}.joblib"))

    if center_and_normalize:
        print("Post-SVD scaling (mean=0, unit-variance per dim) and L2 normalization...")
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr_red = scaler.fit_transform(Xtr_red)
        Xte_red = scaler.transform(Xte_red)
        # L2 normalize rows to stabilize downstream linear models
        normalizer = Normalizer(norm="l2")
        Xtr_red = normalizer.transform(Xtr_red)
        Xte_red = normalizer.transform(Xte_red)
    else:
        # Still cast to float32 to save RAM
        Xtr_red = Xtr_red.astype(np.float32, copy=False)
        Xte_red = Xte_red.astype(np.float32, copy=False)

    return Xtr_red, Xte_red


def attach_and_save(
    train_ids: pd.Series,
    test_ids: pd.Series,
    Xtr_red: np.ndarray,
    Xte_red: np.ndarray,
    vlm_train_csv: str,
    vlm_test_csv: str,
    output_dir: str,
    prefix: str,
    save_format: str = "parquet",
):
    os.makedirs(output_dir, exist_ok=True)
    cols = [f"{prefix}_{i:04d}" for i in range(Xtr_red.shape[1])]
    df_tr = pd.DataFrame(Xtr_red, columns=cols)
    df_tr.insert(0, "sample_id", train_ids.values)
    df_te = pd.DataFrame(Xte_red, columns=cols)
    df_te.insert(0, "sample_id", test_ids.values)

    # Merge with VLM structured data if present
    def _merge(vlm_path: str, feat_df: pd.DataFrame) -> pd.DataFrame:
        if os.path.exists(vlm_path):
            base = pd.read_csv(vlm_path)
            merged = base.merge(feat_df, on="sample_id", how="left")
            return merged
        else:
            print(f"Warning: {vlm_path} not found. Saving TF-IDF features only.")
            return feat_df

    merged_tr = _merge(vlm_train_csv, df_tr)
    merged_te = _merge(vlm_test_csv, df_te)

    out_tr = os.path.join(output_dir, f"vlm_plus_{prefix}_train.{ 'parquet' if save_format=='parquet' else 'csv'}")
    out_te = os.path.join(output_dir, f"vlm_plus_{prefix}_test.{ 'parquet' if save_format=='parquet' else 'csv'}")

    if save_format == "parquet":
        merged_tr.to_parquet(out_tr, index=False)
        merged_te.to_parquet(out_te, index=False)
    else:
        merged_tr.to_csv(out_tr, index=False)
        merged_te.to_csv(out_te, index=False)

    print(f"Saved: {out_tr} ({len(merged_tr)} rows, {merged_tr.shape[1]} cols)")
    print(f"Saved: {out_te} ({len(merged_te)} rows, {merged_te.shape[1]} cols)")


def main():
    parser = argparse.ArgumentParser(description="Build TF-IDF (word + char) with optional SVD, then attach to VLM CSVs")
    parser.add_argument("--train_csv", default="dataset/train.csv")
    parser.add_argument("--test_csv", default="dataset/test.csv")
    parser.add_argument("--text_col", default="catalog_content")
    parser.add_argument("--vlm_train_csv", default="output/vlm_structured_data_train.csv")
    parser.add_argument("--vlm_test_csv", default="output/vlm_structured_data_test.csv")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--artifacts_dir", default="artifacts/tfidf")
    parser.add_argument("--prefix", default="tfidf_svd")
    parser.add_argument("--save_format", choices=["parquet", "csv"], default="parquet")

    # Vectorizer caps
    parser.add_argument("--max_features_word", type=int, default=200_000)
    parser.add_argument("--max_features_char", type=int, default=300_000)
    parser.add_argument("--min_df_word", type=int, default=3)
    parser.add_argument("--max_df_word", type=float, default=0.98)
    parser.add_argument("--min_df_char", type=int, default=3)
    parser.add_argument("--max_df_char", type=float, default=1.0)

    # SVD
    parser.add_argument("--svd_components", type=int, default=4096, help="Set between 2000 and 8000 typically")
    parser.add_argument("--no_svd", action="store_true", help="If set, do not run SVD; will not attach enormous sparse TF-IDF to CSV")
    parser.add_argument("--svd_iter", type=int, default=7)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--no_center", action="store_true", help="Disable StandardScaler/Normalizer after SVD")

    args = parser.parse_args()

    df_tr_all = _read_csv_safe(args.train_csv)
    df_te_all = _read_csv_safe(args.test_csv)

    if args.text_col not in df_tr_all.columns:
        raise KeyError(f"Column '{args.text_col}' not found in train CSV")
    if args.text_col not in df_te_all.columns:
        raise KeyError(f"Column '{args.text_col}' not found in test CSV")

    # Keep only required columns for memory efficiency
    df_tr = df_tr_all[["sample_id", args.text_col]].copy()
    df_te = df_te_all[["sample_id", args.text_col]].copy()
    print(f"Loaded train={len(df_tr)} test={len(df_te)}")

    Xtr, Xte, word_vec, char_vec = build_tfidf_matrices(
        train_text=df_tr[args.text_col],
        test_text=df_te[args.text_col],
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
        max_features_word=args.max_features_word,
        max_features_char=args.max_features_char,
        min_df_word=args.min_df_word,
        max_df_word=args.max_df_word,
        min_df_char=args.min_df_char,
        max_df_char=args.max_df_char,
        artifacts_dir=args.artifacts_dir,
    )

    if args.no_svd:
        raise RuntimeError(
            "SVD is disabled but attaching raw sparse TF-IDF to CSV is not supported in this script. "
            "Please enable SVD (recommended 2000-8000 components) to produce compact features."
        )

    Xtr_red, Xte_red = reduce_with_svd(
        Xtr=Xtr,
        Xte=Xte,
        n_components=args.svd_components,
        random_state=args.random_state,
        n_iter=args.svd_iter,
        center_and_normalize=not args.no_center,
        artifacts_dir=args.artifacts_dir,
    )

    # Free large sparse matrices after reduction
    del Xtr, Xte
    gc.collect()

    attach_and_save(
        train_ids=df_tr["sample_id"],
        test_ids=df_te["sample_id"],
        Xtr_red=Xtr_red,
        Xte_red=Xte_red,
        vlm_train_csv=args.vlm_train_csv,
        vlm_test_csv=args.vlm_test_csv,
        output_dir=args.output_dir,
        prefix=args.prefix,
        save_format=args.save_format,
    )


if __name__ == "__main__":
    main()



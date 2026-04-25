from rag.bm25_store import BM25Store


def test_bm25_retrieves_exact_term_match():
    docs = [
        "The budget allocates funds to education.",
        "NHIL was revised in 2024.",
        "The election results in Ayawaso West favoured NPP.",
    ]
    meta = [{"id": i} for i in range(3)]
    store = BM25Store.build(docs, meta)
    hits = store.top_k("NHIL", k=2)
    assert hits[0]["id"] == 1
    assert hits[0]["score"] > hits[1]["score"]


def test_bm25_save_load_roundtrip(tmp_path):
    docs = ["apple banana", "banana cherry", "cherry date"]
    meta = [{"id": i} for i in range(3)]
    store = BM25Store.build(docs, meta)
    path = tmp_path / "bm25.pkl"
    store.save(path)
    loaded = BM25Store.load(path)
    a = [h["id"] for h in store.top_k("banana", k=3)]
    b = [h["id"] for h in loaded.top_k("banana", k=3)]
    assert a == b

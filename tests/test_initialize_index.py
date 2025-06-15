import os
import ast
from pathlib import Path
from unittest import mock
import tempfile

class DummyState(dict):
    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value


class DummySt:
    def __init__(self):
        self.session_state = DummyState()
    def cache_resource(self, func=None, **kwargs):
        def decorator(f):
            return f
        return decorator if func is None else func

class DummyDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

class DummyTextLoader:
    def __init__(self, path, encoding="utf-8"):
        self.path = path
        self.encoding = encoding
    def load(self):
        content = Path(self.path).read_text(encoding=self.encoding)
        return [DummyDocument(content, {"source": self.path})]

class DummySplitter:
    @classmethod
    def from_tiktoken_encoder(cls, chunk_size=0, chunk_overlap=0):
        return cls()
    def split_documents(self, docs):
        return docs

class DummyChroma:
    def __init__(self, persist_directory=None, embedding_function=None, collection_name=None):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.docs = []
    @classmethod
    def from_documents(cls, docs, emb, persist_directory=None, collection_name=None):
        return cls(persist_directory, emb, collection_name)
    def get(self):
        return {"metadatas": []}
    def add_documents(self, docs):
        self.docs.extend(docs)
    def persist(self):
        pass
    def as_retriever(self):
        ret = mock.MagicMock()
        ret.vectorstore = self
        return ret


def load_initialize_index(st):
    src = Path("adaptive_rag_streamlit_app.py").read_text()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "initialize_index")
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    env = {
        "os": os,
        "glob": __import__("glob"),
        "Path": Path,
        "NomicEmbeddings": None,
        "Chroma": None,
        "PyPDFLoader": DummyTextLoader,
        "TextLoader": DummyTextLoader,
        "RecursiveCharacterTextSplitter": DummySplitter,
        "time": __import__("time"),
        "st": st,
    }
    exec(compile(module, "init", "exec"), env)
    return env["initialize_index"]


def test_initialize_index_rebuild(tmp_path):
    docs_dir = tmp_path / "docs"
    db_dir = tmp_path / "db"
    docs_dir.mkdir()
    db_dir.mkdir()
    sample = docs_dir / "sample.txt"
    sample.write_text("hello")

    st = DummySt()
    init_fn = load_initialize_index(st)

    dummy_file = db_dir / "old.txt"
    dummy_file.write_text("data")

    with (
        mock.patch.dict(
            init_fn.__globals__,
            {"NomicEmbeddings": mock.MagicMock(return_value="emb"), "Chroma": DummyChroma},
        ),
        mock.patch("time.strftime", side_effect=["t1", "t2"]),
    ):
        init_fn(str(docs_dir), str(db_dir))
        first_ts = st.session_state.get("last_index_update")
        assert first_ts == "t1"

        assert dummy_file.exists()
        init_fn(str(docs_dir), str(db_dir), rebuild=True)
        second_ts = st.session_state.get("last_index_update")

    assert second_ts == "t2"
    assert not dummy_file.exists()
    assert db_dir.exists()

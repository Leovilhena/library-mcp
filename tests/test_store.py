from pathlib import Path

from library_mcp.store import add_book, add_chunk, commit, open_store, search


def test_search_ranks_by_similarity_across_books(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    net_book = add_book(conn, "Networking", "/inbox/net.pdf", "2026-01-01")
    add_chunk(conn, net_book, 0, "p1", "python asyncio for networking", [1.0, 0.0, 0.0])
    add_chunk(conn, net_book, 1, "p2", "TCP handshake basics", [0.0, 1.0, 0.0])

    py_book = add_book(conn, "Python", "/inbox/py.pdf", "2026-01-01")
    add_chunk(conn, py_book, 0, "ch1", "python asyncio event loops", [0.99, 0.01, 0.0])
    commit(conn)

    results = search(conn, [1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    titles = {r.book_title for r in results}
    assert titles == {"Networking", "Python"}
    assert results[0].score >= results[1].score


def test_search_excludes_irrelevant_chunks_when_top_k_is_tight(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    book = add_book(conn, "Book", "/inbox/b.pdf", "2026-01-01")
    add_chunk(conn, book, 0, "p1", "relevant", [1.0, 0.0])
    add_chunk(conn, book, 1, "p2", "irrelevant", [0.0, 1.0])
    commit(conn)

    results = search(conn, [1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].text == "relevant"

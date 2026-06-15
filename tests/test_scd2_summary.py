"""
Unit tests for ``SCD2DeltaTableWriter._summarize_merge_actions``.

These tests deliberately live outside ``tests/spark`` and do **not** use the ``spark`` fixture.
The ``tests/spark`` package has a session-scoped, autouse ``setup`` fixture that starts a
SparkSession, so any test collected there requires a JVM / Delta runtime. The summary-field
computation, however, is pure Python, so it can be verified locally without Spark or Delta:

    PYTHONPATH=src python -m pytest tests/test_scd2_summary.py -q

The end-to-end behavior (real Delta merge + the summary exposed on ``writer.output.summary`` and
logged) is covered separately by ``tests/spark/writers/delta/test_scd.py::test_scd2_summary``,
which runs where a Spark/Delta runtime is available (e.g. CI).
"""

from koheesio.spark.writers.delta.scd import SCD2DeltaTableWriter


def test_summarize_merge_actions_inserts_only():
    """Only brand new dimension members -> inserts; nothing updated or closed."""
    assert SCD2DeltaTableWriter._summarize_merge_actions([("I", 1, 3)], source_rows=3) == {
        "new_records": 3,
        "updated_records": 0,
        "closed_records": 0,
        "scd1_updates": 0,
        "total_affected_records": 3,
        "source_rows": 3,
        "unchanged_records": 0,
    }


def test_summarize_merge_actions_mixed_batch():
    """1 insert, 1 SCD2 change (UC rn=1 new version + rn=2 closed history), 1 SCD1 in-place update.

    7 of the 10 source rows were unchanged (10 - (1 new + 1 updated + 1 scd1)).
    """
    assert SCD2DeltaTableWriter._summarize_merge_actions(
        [("I", 1, 1), ("UC", 1, 1), ("UC", 2, 1), ("U", 1, 1)],
        source_rows=10,
    ) == {
        "new_records": 1,
        "updated_records": 1,
        "closed_records": 1,
        "scd1_updates": 1,
        "total_affected_records": 4,
        "source_rows": 10,
        "unchanged_records": 7,
    }


def test_summarize_merge_actions_empty():
    """No rows staged (nothing changed) -> every counter is zero."""
    assert SCD2DeltaTableWriter._summarize_merge_actions([]) == {
        "new_records": 0,
        "updated_records": 0,
        "closed_records": 0,
        "scd1_updates": 0,
        "total_affected_records": 0,
    }


def test_summarize_merge_actions_without_source_rows():
    """When source_rows is omitted, source/unchanged keys are not added.

    A UC pair contributes one updated record (rn=1) and one closed record (rn=2) per merge key.
    """
    assert SCD2DeltaTableWriter._summarize_merge_actions([("UC", 1, 2), ("UC", 2, 2)]) == {
        "new_records": 0,
        "updated_records": 2,
        "closed_records": 2,
        "scd1_updates": 0,
        "total_affected_records": 4,
    }

"""
Test the CleanseStrings columns transformation
"""

import pytest

from pydantic import ValidationError

from koheesio.logger import LoggingFactory
from koheesio.spark.transformations.strings.change_case import LowerCase, TitleCase, UpperCase
from koheesio.spark.transformations.strings.clean import (
    _ACTION_FUNCTIONS,
    CleanseStrings,
    StringCleanupAction,
)
from koheesio.spark.transformations.strings.trim import LTrim, RTrim, Trim
from koheesio.spark.utils import show_string

pytestmark = pytest.mark.spark

log = LoggingFactory.get_logger(name=__file__, inherit_from_koheesio=True)


def _column_values(df, column):
    """Collect a single column into a plain Python list, preserving row order."""
    return [row.asDict()[column] for row in df.collect()]


#
# Multi-column processing
#
@pytest.mark.parametrize(
    "input_values,input_data,input_schema,expected",
    [
        (
            # description: default actions (trim + empty_to_null) applied in place to two explicit columns
            dict(columns=["name", "city"]),
            [[" Alice ", "  "], ["BOB  ", "Delft"], [None, None]],
            ["name", "city"],
            [
                dict(name="Alice", city=None),
                dict(name="BOB", city="Delft"),
                dict(name=None, city=None),
            ],
        ),
        (
            # description: trim + lower into suffixed target columns; sources remain untouched.
            # "  " becomes "" (empty string, NOT null) because empty_to_null is not in the action list.
            dict(columns=["name", "city"], target_column="clean", actions=["trim", "lower"]),
            [[" Alice ", "  "], ["BOB  ", "Delft"], [None, None]],
            ["name", "city"],
            [
                dict(name=" Alice ", city="  ", name_clean="alice", city_clean=""),
                dict(name="BOB  ", city="Delft", name_clean="bob", city_clean="delft"),
                dict(name=None, city=None, name_clean=None, city_clean=None),
            ],
        ),
        (
            # description: default actions (trim + empty_to_null) into suffixed target columns. Null lands in the
            # suffixed columns and the source columns are preserved unchanged. "  " -> trim "" -> empty_to_null None.
            dict(columns=["name", "city"], target_column="clean"),
            [[" Alice ", "  "], ["BOB  ", "Delft"], [None, None]],
            ["name", "city"],
            [
                dict(name=" Alice ", city="  ", name_clean="Alice", city_clean=None),
                dict(name="BOB  ", city="Delft", name_clean="BOB", city_clean="Delft"),
                dict(name=None, city=None, name_clean=None, city_clean=None),
            ],
        ),
    ],
)
def test_multi_column(input_values, input_data, input_schema, expected, spark):
    input_df = spark.createDataFrame(input_data, input_schema)

    output_df = CleanseStrings(**input_values).transform(input_df)

    log.info(f"show output_df: \n{show_string(output_df, 20, 20, False)}")
    actual = [row.asDict() for row in output_df.collect()]
    assert actual == expected


def test_all_string_columns_and_non_string_skipped(spark):
    """Omitting columns cleanses every string column; non-string columns are left untouched."""
    input_df = spark.createDataFrame(
        [[1, " foo ", "  "], [2, "BAR  ", None]], ["id", "input_column", "other_column"]
    )

    # default actions: trim + empty_to_null
    output_df = CleanseStrings().transform(input_df)

    log.info(f"show output_df: \n{show_string(output_df, 20, 20, False)}")
    actual = [row.asDict() for row in output_df.collect()]
    assert actual == [
        dict(id=1, input_column="foo", other_column=None),
        dict(id=2, input_column="BAR", other_column=None),
    ]


#
# Null value handling / preservation
#
@pytest.mark.parametrize(
    "actions,expected",
    [
        (
            # default: trim then empty_to_null -> whitespace-only and empty become null, existing null preserved
            ["trim", "empty_to_null"],
            ["hello", None, None, None],
        ),
        (
            # empty_to_null only: only the literal empty string becomes null; whitespace stays as-is,
            # existing null preserved, non-empty values untouched
            ["empty_to_null"],
            [" hello ", "   ", None, None],
        ),
        (
            # trim only: no value is turned into null; existing null is preserved
            ["trim"],
            ["hello", "", "", None],
        ),
    ],
)
def test_null_handling(actions, expected, spark):
    input_df = spark.createDataFrame([[" hello "], ["   "], [""], [None]], ["val"])

    output_df = CleanseStrings(column="val", actions=actions).transform(input_df)

    log.info(f"show output_df: \n{show_string(output_df, 20, 20, False)}")
    assert _column_values(output_df, "val") == expected


def test_action_order_is_significant(spark):
    """Order matters: running empty_to_null *before* trim leaves whitespace-only values as empty strings,
    whereas the default trim-then-empty_to_null turns them into null."""
    input_df = spark.createDataFrame([[" hello "], ["   "], [""], [None]], ["val"])

    empty_then_trim = CleanseStrings(column="val", actions=["empty_to_null", "trim"]).transform(input_df)
    trim_then_empty = CleanseStrings(column="val", actions=["trim", "empty_to_null"]).transform(input_df)

    # empty_to_null runs first: only the literal "" becomes null; "   " is not empty yet, so it survives the
    # null conversion and is only afterwards trimmed down to "".
    assert _column_values(empty_then_trim, "val") == ["hello", "", None, None]
    # default order: trim first turns "   " into "", which empty_to_null then converts to null.
    assert _column_values(trim_then_empty, "val") == ["hello", None, None, None]


#
# Single-column behavior matches the dedicated transformations (backward compatibility)
#
@pytest.mark.parametrize(
    "action,sibling_cls",
    [
        ("trim", Trim),
        ("ltrim", LTrim),
        ("rtrim", RTrim),
        ("lower", LowerCase),
        ("upper", UpperCase),
        ("title", TitleCase),
    ],
)
def test_single_column_matches_sibling_in_place(action, sibling_cls, spark):
    """A single action on a single column (in place) yields the same result as the dedicated transformation."""
    input_df = spark.createDataFrame([[" Foo Bar "], ["BAZ"], [None]], ["name"])

    cleansed = CleanseStrings(column="name", actions=action).transform(input_df)
    reference = sibling_cls(column="name").transform(input_df)

    log.info(f"show cleansed: \n{show_string(cleansed, 20, 20, False)}")
    assert _column_values(cleansed, "name") == _column_values(reference, "name")


def test_single_column_matches_trim_with_target(spark):
    """Single column with a target_column writes to the target, identical to Trim with a target_column."""
    input_df = spark.createDataFrame([[" Foo "], ["bar  "], [None]], ["name"])

    cleansed = CleanseStrings(column="name", target_column="out", actions="trim").transform(input_df)
    reference = Trim(column="name", target_column="out").transform(input_df)

    assert _column_values(cleansed, "out") == _column_values(reference, "out")
    # source column is preserved unchanged
    assert _column_values(cleansed, "name") == _column_values(reference, "name")


#
# Case standardization combined with trimming
#
def test_trim_then_upper(spark):
    input_df = spark.createDataFrame([[" mixed Case "], [None]], ["val"])

    output_df = CleanseStrings(column="val", actions=["trim", "upper"]).transform(input_df)

    assert _column_values(output_df, "val") == ["MIXED CASE", None]


#
# Validation / configuration
#
def test_all_actions_have_dispatch():
    """Every declared cleanup action must have a corresponding column-expression in the dispatch table."""
    assert set(StringCleanupAction) == set(_ACTION_FUNCTIONS)


def test_default_actions(spark):
    """The default action combination is trim followed by empty_to_null."""
    transform = CleanseStrings(column="name")
    assert transform.actions == [StringCleanupAction.TRIM, StringCleanupAction.EMPTY_TO_NULL]


def test_single_action_string_is_coerced_to_list(spark):
    """A single action (as a string) is accepted and coerced into a one-element list."""
    transform = CleanseStrings(column="name", action="trim")
    assert transform.actions == [StringCleanupAction.TRIM]


def test_actions_alias(spark):
    """The `action` alias populates the `actions` field."""
    transform = CleanseStrings(column="name", action=["trim", "lower"])
    assert transform.actions == [StringCleanupAction.TRIM, StringCleanupAction.LOWER]


def test_invalid_action_raises():
    with pytest.raises(ValidationError):
        CleanseStrings(column="name", action="frobnicate")

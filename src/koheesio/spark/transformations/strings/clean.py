"""
Cleanse the contents of one or more string columns by applying a configurable, ordered combination of cleaning
actions in a single pass.

This module builds on the single-purpose string transformations in this package (`trim`, `change_case`, `replace`)
and is intended for the common data-cleaning scenario where several columns need the same set of normalisations at
once - for example: trim surrounding whitespace, convert resulting empty strings to `null`, and standardise case.

Classes
-------
- `CleanseStrings`
    Apply an ordered list of cleaning actions (trim, empty-string-to-null, case standardisation, ...) to one or more
    string columns.

See the class docstring for more information.
"""

from typing import Callable, Dict, List, Union
from enum import Enum

from pyspark.sql import Column
import pyspark.sql.functions as f

from koheesio.models import Field, ListOfColumns, field_validator
from koheesio.spark.transformations import ColumnsTransformationWithTarget
from koheesio.spark.utils import SparkDatatype

__all__ = ["CleanseStrings", "StringCleanupAction"]


class StringCleanupAction(str, Enum):
    """The cleaning actions supported by `CleanseStrings`.

    Each action maps to a column expression that mirrors the behaviour of the dedicated transformations in this
    package, so that combining them here produces the same result as chaining the individual transformations.

    Members
    -------
    TRIM : "trim"
        Trim whitespace from both sides. Equivalent to `Trim(direction="left-right")`.
    LTRIM : "ltrim"
        Trim whitespace from the left side. Equivalent to `LTrim`.
    RTRIM : "rtrim"
        Trim whitespace from the right side. Equivalent to `RTrim`.
    LOWER : "lower"
        Convert to lower case. Equivalent to `LowerCase`.
    UPPER : "upper"
        Convert to upper case. Equivalent to `UpperCase`.
    TITLE : "title"
        Convert to title case. Equivalent to `TitleCase` / `InitCap`.
    EMPTY_TO_NULL : "empty_to_null"
        Replace empty strings (`""`) with `null`. Comparable to using `Replace` to map a value to `null`, but
        targeted at the empty string specifically. Existing `null` values are left untouched.
    """

    TRIM = "trim"
    LTRIM = "ltrim"
    RTRIM = "rtrim"
    LOWER = "lower"
    UPPER = "upper"
    TITLE = "title"
    EMPTY_TO_NULL = "empty_to_null"


# Mapping of each action to the column expression that implements it.
#
# Null preservation: every PySpark string function below returns `null` when its input is `null`, and the
# `empty_to_null` branch falls through to `otherwise(column)` for a `null` input (because `null == ""` evaluates to
# `null`, not `true`). As a result, pre-existing `null` values are preserved by every action.
_ACTION_FUNCTIONS: Dict[StringCleanupAction, Callable[[Column], Column]] = {
    StringCleanupAction.TRIM: lambda column: f.rtrim(f.ltrim(column)),
    StringCleanupAction.LTRIM: f.ltrim,
    StringCleanupAction.RTRIM: f.rtrim,
    StringCleanupAction.LOWER: f.lower,
    StringCleanupAction.UPPER: f.upper,
    StringCleanupAction.TITLE: f.initcap,
    StringCleanupAction.EMPTY_TO_NULL: lambda column: f.when(
        column == "", f.lit(None).cast("string")
    ).otherwise(column),
}


class CleanseStrings(ColumnsTransformationWithTarget):
    """Cleanse one or more string columns by applying an ordered combination of cleaning actions in a single pass.

    This is a multi-column convenience transformation built on top of the single-purpose string transformations in
    this package. Instead of chaining `Trim`, `LowerCase`, `Replace`, etc., a user configures a list of `columns` and
    a list of `actions`; each action is applied to every column, in the order given.

    Because it extends `ColumnsTransformationWithTarget`, all the standard multi-column behaviour applies:

    - the actions are run in a loop against every column in `columns`;
    - if a single column is given, the result is written in place, or to `target_column` if provided - identical to
      the behaviour of the individual string transformations (e.g. `Trim`);
    - if multiple columns are given and `target_column` is set, it is used as a *suffix* so each source column gets
      its own result column (`<column>_<target_column>`);
    - if `columns` is omitted (or `"*"`), all string columns are cleansed.

    Null handling
    -------------
    Existing `null` values are always preserved - none of the actions turn a `null` into a non-null value. The
    `empty_to_null` action additionally turns empty strings (`""`) into `null`. To treat whitespace-only values as
    empty, place a trim action before `empty_to_null` (e.g. `["trim", "empty_to_null"]`), which is the default.

    Action order
    ------------
    Actions are applied left-to-right, so the order is significant. For example `["trim", "empty_to_null", "lower"]`
    trims whitespace, then converts the now-empty strings to `null`, then lower-cases the remainder.

    Warnings
    --------
    If the type of a column is not string, it will be skipped and a warning will be thrown - consistent with the other
    string transformations.

    Parameters
    ----------
    columns : ListOfColumns, optional, default="*"
        The column (or list of columns) to cleanse. Alias: column. If not provided, all string columns are cleansed.
    target_column : Optional[str], optional, default=None
        The column to store the result in. If not provided, the result is stored in the source column. Alias:
        target_suffix - if multiple columns are given as source, this is used as a suffix instead.
    actions : Union[StringCleanupAction, List[StringCleanupAction]], optional, default=["trim", "empty_to_null"]
        An ordered list of cleaning actions to apply to each column. A single action may be passed instead of a list.
        Supported actions: "trim", "ltrim", "rtrim", "lower", "upper", "title", "empty_to_null". Alias: action.

    Examples
    --------
    __input_df:__

    | name      | city    |
    |-----------|---------|
    | " Alice " | "  "    |
    | "BOB  "   | "Delft" |
    | None      | None    |

    ### Trim and convert empty strings to null across multiple columns (default actions)
    ```python
    output_df = CleanseStrings(columns=["name", "city"]).transform(input_df)
    ```

    __output_df:__

    | name    | city    |
    |---------|---------|
    | "Alice" | None    |
    | "BOB"   | "Delft" |
    | None    | None    |

    ### Trim and lower-case into new suffixed columns
    ```python
    output_df = CleanseStrings(
        columns=["name", "city"],
        target_column="clean",
        actions=["trim", "lower"],
    ).transform(input_df)
    ```

    __output_df:__

    | name      | city    | name_clean | city_clean |
    |-----------|---------|------------|------------|
    | " Alice " | "  "    | "alice"    | ""         |
    | "BOB  "   | "Delft" | "bob"      | "delft"    |
    | None      | None    | None       | None       |

    ### Single column, single action - identical to using `Trim`
    ```python
    output_df = CleanseStrings(column="name", actions="trim").transform(input_df)
    ```
    """

    class ColumnConfig(ColumnsTransformationWithTarget.ColumnConfig):
        """Limit data types to string only."""

        run_for_all_data_type = [SparkDatatype.STRING]
        limit_data_type = [SparkDatatype.STRING]

    columns: ListOfColumns = Field(
        default="*",
        alias="column",
        description="The column (or list of columns) to cleanse. Alias: column. If no columns are provided, all "
        "string columns will be cleansed.",
    )
    actions: List[StringCleanupAction] = Field(
        default=[StringCleanupAction.TRIM, StringCleanupAction.EMPTY_TO_NULL],
        alias="action",
        description="An ordered list of cleaning actions to apply to each column. A single action may be passed "
        "instead of a list. Supported: 'trim', 'ltrim', 'rtrim', 'lower', 'upper', 'title', 'empty_to_null'.",
    )

    @field_validator("actions", mode="before")
    def _coerce_actions_to_list(cls, value: Union[str, StringCleanupAction, List]) -> List:
        """Allow a single action to be passed instead of a list, mirroring the ergonomics of `ListOfColumns`."""
        if value is None:
            return value
        if isinstance(value, (str, StringCleanupAction)):
            return [value]
        return value

    def func(self, column: Column) -> Column:
        for action in self.actions:
            column = _ACTION_FUNCTIONS[action](column)
        return column

# Query syntax reference

This page defines everything that can appear in a predicate — the `where=`
argument of {meth}`Table.select <h5col.Table.select>`,
{meth}`Table.read <h5col.Table.read>`, and
{meth}`Table.count <h5col.Table.count>`. The syntax intentionally parallels
pyarrow's, in both its expression form and its tuple form, so predicates
written for one usually read verbatim in the other.

## Expressions

{func}`~h5col.field` names a column and supports the comparison operators
directly; each comparison yields an {class}`~h5col.Expression`:

| Predicate | Meaning |
|---|---|
| `field("x") == v` | equal to `v` |
| `field("x") != v` | not equal to `v` |
| `field("x") < v`, `<= v`, `> v`, `>= v` | ordered comparison |
| `field("x").isin(values)` | equal to any element of `values` |
| `field("x").is_null()` | the row's value is missing |
| `field("x").is_valid()` | the row's value is present |

Expressions combine with the logical operators:

| Combinator | Meaning |
|---|---|
| `a & b` | both |
| `a \| b` | either |
| `~a` | negation |

Python gives `&`, `|`, and `~` higher precedence than comparisons, so each
comparison must be parenthesized — this is a property of the language, and
pyarrow users will already have the habit:

```python
(field("kind") == "automatic") & (field("t_air") > 22.0)   # correct
field("kind") == "automatic" & field("t_air") > 22.0       # wrong: & binds first
```

A bare `field("x")` is not a predicate; passing one raises
{class}`~h5col.SchemaError`.

## Values

A comparison value should live in the column's value domain:

- Numeric columns compare against Python or NumPy numbers. Mixed
  integer/float comparisons are exact — the engine compares in Python's
  object domain, so a `uint64` beyond 2⁵³ is never silently rounded through
  `float64`.
- Fixed-length string columns compare against `str` (or UTF-8 `bytes`);
  ordering is bytewise, which for ASCII and UTF-8 equals codepoint order.
- Boolean columns compare against `True`/`False`.
- Categorical columns compare against labels, never codes —
  `field("payment_type") == "Cash"`. An unknown label under `==` or
  `isin` simply matches nothing; an unknown label under an ordering
  operator raises {class}`~h5col.SchemaError`, since it has no defined
  position. Ordering comparisons on a categorical follow the declared
  category order (the code order), which is chiefly meaningful for columns
  declared with `ordered=True`.
- Datetime columns, stored as integers by an application-level codec (H5Col
  defines no datetime type), compare against the encoded integers — encode
  the query bound with the same codec used to write the column, as the
  {doc}`taxi example <../notebooks/06_nyc_taxi>` demonstrates.

## Tuple form

Anywhere an expression is accepted, pyarrow's tuple filters work as well:

| Form | Meaning |
|---|---|
| `("x", op, v)` | one predicate |
| `[t1, t2, ...]` | AND of the tuples |
| `[[t1, t2], [t3], ...]` | OR of AND-groups (disjunctive normal form) |

The operator token is a string: `"="` or `"=="`, `"!="`, `"<"`, `"<="`,
`">"`, `">="`, `"in"`, or `"not in"` (`"in"` takes a sequence of values).
The two spellings below are the same query:

```python
table.select((field("kind") == "automatic") & (field("t_air") > 22.0))
table.select([("kind", "==", "automatic"), ("t_air", ">", 22.0)])
```

An empty list selects every row. (pyarrow, by contrast, rejects an empty
`filters` list as malformed — this is one small place the two APIs differ.)

## Missing values: three-valued logic

Comparisons involving a missing value are neither true nor false — they are
unknown, and a row is selected only when its whole predicate evaluates to
true. The connectives follow Kleene logic, the same rules SQL and pyarrow
apply:

| `a` | `b` | `a & b` | `a \| b` | `~a` |
|---|---|---|---|---|
| true | unknown | unknown | true | false |
| false | unknown | false | unknown | true |
| unknown | unknown | unknown | unknown | unknown |

The practical consequences:

- `field("x") == 5` never matches a missing row — and neither does
  `~(field("x") == 5)` or `field("x") != 5`. Negation does not turn unknown
  into true.
- The only predicates that see missing rows are `is_null()` and
  `is_valid()`. They are always definite (a row is either present or not),
  and they negate into each other: `~field("x").is_null()` is exactly
  `field("x").is_valid()`.
- To include missing rows in an otherwise value-based selection, say so:
  `(field("x") > 5) | field("x").is_null()`.

Boolean columns cannot hold missing values, so their comparisons are always
definite.

## Errors and limits

- A structurally malformed `where=` value — an unknown operator token, a
  shape that is neither an expression nor tuples — raises
  {class}`~h5col.SchemaError` immediately, from `select()`, `read()`, or
  `count()` itself.
- A predicate naming a column the table does not have raises
  {class}`KeyError` when the selection is first evaluated; predicates on
  [list columns](../guide/list-columns.md) are not supported and raise
  {class}`~h5col.SchemaError` at the same point.
- Distributing AND over OR during normalization caps at 1,024 AND-terms; a
  predicate that expands beyond that (deeply nested negated conjunctions
  can) is rejected with {class}`~h5col.SchemaError` and should be
  simplified. A filter already written as a list of OR-groups is not
  subject to the cap.

## Correspondence with pyarrow

| pyarrow | h5col |
|---|---|
| `pc.field("x") == v`, `!=`, `<`, `<=`, `>`, `>=` | `field("x") == v`, ... (same operators) |
| `pc.field("x").isin(values)` | `field("x").isin(values)` |
| `pc.field("x").is_null()` / `.is_valid()` | `field("x").is_null()` / `.is_valid()` |
| `&`, `\|`, `~` on expressions | `&`, `\|`, `~` on expressions |
| `filters=[("x", "in", {...}), ...]` | the same tuples, passed to `select()` |
| null semantics in filters (Kleene) | the same three-valued semantics |

The differences are the import — `field` comes from `h5col` — and the value
domains noted above (labels for categoricals, encoded integers for
datetimes).

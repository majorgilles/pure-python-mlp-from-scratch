# AGENTS.md

## Project conventions

- Use explicit Python type hints for new or changed project code.
- Prefer precise standard-library types over untyped `dict`, `list`, or `Any` when the value shape is known.
- When a structured data object needs validation, parsing, or serialization, prefer a Pydantic `BaseModel` instead of a `dataclass`.
- Use simple built-in types or type aliases when validation is unnecessary.

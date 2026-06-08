# AGENTS.md

## Project conventions

- Use explicit Python type hints for new or changed project code when they clarify interfaces, structured values, or non-obvious shapes.
- Do not add type annotations for simple local variables in scripts or notebooks when the assigned value makes the type obvious, such as `seed = 7`, `count = 5`, or `total = 0.0`.
- Prefer precise standard-library types over untyped `dict`, `list`, or `Any` when the value shape is known.
- When a structured data object needs validation, parsing, or serialization, prefer a Pydantic `BaseModel` instead of a `dataclass`.
- Use simple built-in types or type aliases when validation is unnecessary.
- In learner-facing notebooks and docs, avoid scientific notation for numeric examples; format small floats as readable decimals unless scientific notation is the lesson topic.

"""Strict tool schemas and actionable, source-sized argument diagnostics."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


def tool_parameter_schema(schema: dict[str, Any], *, require_all: bool = True) -> dict[str, Any]:
    """Close object arguments recursively; the wire protocol requires every named field."""
    result = deepcopy(schema)

    def normalize(node: dict[str, Any]) -> None:
        if node.get("type") == "object" or "properties" in node:
            node.setdefault("additionalProperties", False)
            if require_all:
                node["required"] = list(node.get("properties", {}))
        for key in ("properties", "$defs", "definitions", "patternProperties", "dependentSchemas"):
            for child in node.get(key, {}).values():
                if isinstance(child, dict):
                    normalize(child)
        for key in (
            "items",
            "contains",
            "additionalProperties",
            "not",
            "if",
            "then",
            "else",
            "propertyNames",
        ):
            child = node.get(key)
            if isinstance(child, dict):
                normalize(child)
        for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
            for child in node.get(key, []):
                if isinstance(child, dict):
                    normalize(child)

    normalize(result)
    return result


class ToolArgumentsError(ValueError):
    """An argument rejection before tool execution, safe to send back to the model."""

    def __init__(
        self,
        tool: str,
        kind: str,
        errors: list[dict[str, Any]],
        *,
        required: list[str],
        provided: list[str],
    ) -> None:
        self.feedback = {
            "tool": tool,
            "errors": errors,
            "required_fields": required,
            "provided_fields": provided,
            "executed": False,
        }
        self.suggestion = "\n".join(error["suggestion"] for error in errors)
        super().__init__(f"error: {kind}: " + json.dumps(self.feedback, ensure_ascii=False))


def _schema_errors(tool: str, error: ValidationError) -> list[dict[str, Any]]:
    path = list(error.absolute_path)
    rule = error.validator
    expected = error.validator_value
    failures: list[tuple[list[str | int], str, Any, dict[str, Any]]] = [
        (path, rule, expected, error.schema)
    ]
    if rule == "required":
        failures = [
            ([*path, key], "missing", key, error.schema.get("properties", {}).get(key, {}))
            for key in expected
            if key not in error.instance
        ]
    elif rule == "additionalProperties":
        failures = [
            ([*path, key], "extra_forbidden", list(error.schema.get("properties", {})), {})
            for key in error.instance
            if key not in error.schema.get("properties", {})
        ]
    results = []
    for location, code, value, field_schema in failures:
        field = ".".join(map(str, location)) or "$"
        actions = {
            "missing": f"Add `{field}` as {field_schema.get('type', 'the declared JSON value')}",
            "extra_forbidden": f"Remove `{field}`; allowed fields here: {value}",
            "type": f"Set `{field}` to JSON type {value}; do not coerce or JSON-encode its value",
            "enum": f"Set `{field}` to one of {json.dumps(value, ensure_ascii=False)}",
            "const": f"Set `{field}` to {json.dumps(value, ensure_ascii=False)}",
            "minimum": f"Set `{field}` >= {value}",
            "maximum": f"Set `{field}` <= {value}",
            "exclusiveMinimum": f"Set `{field}` > {value}",
            "exclusiveMaximum": f"Set `{field}` < {value}",
            "minLength": f"Set `{field}` to a string with at least {value} characters",
            "maxLength": f"Set `{field}` to a string with at most {value} characters",
            "minItems": f"Provide at least {value} items in `{field}`",
            "maxItems": f"Provide at most {value} items in `{field}`",
            "uniqueItems": f"Remove duplicate items from `{field}`",
            "multipleOf": f"Set `{field}` to a multiple of {value}",
            "pattern": f"Set `{field}` to a string matching regex {value!r}",
        }
        if code in {"anyOf", "oneOf"} and error.context:
            alternatives: dict[int, list[str]] = {}
            for child in error.context:
                branch = child.schema_path[0]
                alternatives.setdefault(branch, []).extend(
                    failure["msg"] for failure in _schema_errors(tool, child)
                )
            action = f"For `{field}`, choose one valid alternative: " + " OR ".join(
                "(" + "; ".join(messages) + ")" for messages in alternatives.values()
            )
        else:
            action = actions.get(
                code,
                f"Replace `{field}` using its declared `{code}` constraint: {json.dumps(value, ensure_ascii=False)}",
            )
        if description := field_schema.get("description"):
            action += f". Field requirement: {description}"
        results.append(
            {
                "loc": location,
                "type": code,
                "msg": action,
                "suggestion": f"{tool}: {action}. Retry this tool with the corrected argument object.",
            }
        )
    return results


def validate_tool_payload(
    tool: str, schema: dict[str, Any], arguments: str, *, defaults: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate the original JSON without type coercion, dropped keys or handler side effects."""
    required = schema.get("required", [])

    def object_value(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate argument key {key!r}; keep this key exactly once")
            value[key] = item
        return value

    def finite_number(value: str) -> None:
        raise ValueError(f"{value} is not a JSON number; supply a finite number or an allowed null")

    try:
        payload = json.loads(
            arguments, object_pairs_hook=object_value, parse_constant=finite_number
        )
    except json.JSONDecodeError as error:
        raise ToolArgumentsError(
            tool,
            "invalid JSON arguments",
            [
                {
                    "loc": [],
                    "type": "json_invalid",
                    "msg": error.msg,
                    "suggestion": f"{tool}: Fix JSON at line {error.lineno}, column {error.colno} (offset {error.pos}): {error.msg}. Send one complete argument object; escape string quotes and newlines once.",
                }
            ],
            required=required,
            provided=[],
        ) from error
    except ValueError as error:
        raise ToolArgumentsError(
            tool,
            "invalid JSON arguments",
            [
                {
                    "loc": [],
                    "type": "json_invalid",
                    "msg": str(error),
                    "suggestion": f"{tool}: {error}. Resend the complete argument object.",
                }
            ],
            required=required,
            provided=[],
        ) from error
    if not isinstance(payload, dict):
        raise ToolArgumentsError(
            tool,
            "tool arguments must be a JSON object",
            [
                {
                    "loc": [],
                    "type": "object_type",
                    "msg": "arguments must be an object",
                    "suggestion": f"{tool}: Return a JSON object with named arguments {list(schema.get('properties', {}))}; do not wrap it in a string or array.",
                }
            ],
            required=required,
            provided=[],
        )

    def with_defaults(value: Any, fallback: Any) -> Any:
        if isinstance(value, dict) and isinstance(fallback, dict):
            return {
                key: with_defaults(value[key], fallback.get(key)) if key in value else fallback[key]
                for key in value.keys() | fallback.keys()
            }
        if isinstance(value, list) and isinstance(fallback, list):
            return [
                with_defaults(item, fallback[index]) if index < len(fallback) else item
                for index, item in enumerate(value)
            ]
        return value

    candidate = with_defaults(payload, defaults) if defaults is not None else payload
    errors = []
    seen: set[tuple[tuple[str | int, ...], str]] = set()
    for error in Draft202012Validator(tool_parameter_schema(schema, require_all=False)).iter_errors(
        candidate
    ):
        for failure in _schema_errors(tool, error):
            key = (tuple(failure["loc"]), failure["type"])
            if key not in seen:
                errors.append(failure)
                seen.add(key)
    if errors:
        raise ToolArgumentsError(
            tool, "invalid arguments", errors, required=required, provided=sorted(payload)
        )
    return payload

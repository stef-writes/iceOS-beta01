from typing import Any, Dict, Tuple, Type
import json
from app.utils.type_coercion import coerce_types
from pydantic import BaseModel

class OutputValidationError(Exception):
    """Custom exception for output validation errors."""
    pass

class OutputValidator:
    """
    Validates and coerces the output of an LLM call based on the node's configuration.
    """
    @staticmethod
    def validate_and_coerce(
        generated_text: str,
        output_format: str,
        output_schema: Dict[str, str] | Type[BaseModel]
    ) -> Tuple[Any, bool, str | None]:
        """
        Validates and coerces the generated text based on the specified output format and schema.

        Returns:
            A tuple containing the coerced output, a success flag, and an error message.
        """
        try:
            from app.models.node_models import BaseNodeConfig
            # Handle JSON output
            if output_format == 'json':
                parsed_output = json.loads(generated_text.strip())
                # Support both dict and Pydantic model schemas
                coerced_output = coerce_types(parsed_output, output_schema)
                return coerced_output, True, None
            # For plain text, only coerce if schema is simple (dict with one key) or Pydantic model with one field
            elif output_format == 'plain':
                if BaseNodeConfig.is_pydantic_schema(output_schema):
                    # If Pydantic model with one field, map to that field
                    fields = list(output_schema.model_fields.keys())
                    if len(fields) == 1:
                        key = fields[0]
                        coerced_output = coerce_types({key: generated_text.strip()}, output_schema)
                        return coerced_output, True, None
                    else:
                        return generated_text, False, "Cannot map plain text to complex Pydantic schema. Use JSON output format."
                elif isinstance(output_schema, dict) and len(output_schema) == 1:
                    key = list(output_schema.keys())[0]
                    coerced_output = coerce_types({key: generated_text.strip()}, output_schema)
                    return coerced_output, True, None
                else:
                    return generated_text, False, "Cannot map plain text to complex schema. Use JSON output format."
            # For other formats, return the text directly without coercion
            return generated_text, True, None
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            return None, False, str(e)
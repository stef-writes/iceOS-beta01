from pydantic import BaseModel, ValidationError

def coerce_value(value, target_type):
    try:
        if target_type is int:
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                value = value.replace(",", "").strip()
                if value.lower() in ("nan", "inf", "-inf"):  # edge cases
                    raise ValueError(f"Cannot coerce '{value}' to int")
                if "." in value:
                    return int(float(value))
                return int(value)
        elif target_type is float:
            if isinstance(value, float):
                return value
            if isinstance(value, int):
                return float(value)
            if isinstance(value, str):
                value = value.replace(",", "").strip()
                if value.lower() in ("nan", "inf", "-inf"):
                    raise ValueError(f"Cannot coerce '{value}' to float")
                return float(value)
        elif target_type is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes", "y")
            if isinstance(value, (int, float)):
                return value != 0
        elif target_type is str:
            return str(value)
        else:
            return value
    except Exception:
        raise ValueError(f"Could not coerce value '{value}' to {target_type.__name__}")

def coerce_types(output: dict, schema: any) -> dict:
    """
    Coerces an output dictionary to conform to a given schema, which can be a 
    Pydantic model or a simple type dictionary.
    """
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        try:
            # Pydantic model handles all parsing and validation
            validated_model = schema.model_validate(output)
            return validated_model.model_dump()
        except ValidationError as e:
            raise ValueError(f"Pydantic validation failed: {e}")
    
    # Fallback to original logic for simple dict schemas
    type_map = {"str": str, "int": int, "float": float, "bool": bool}
    coerced = {}
    errors = {}
    
    # Ensure schema is a dictionary for the old logic path
    if not isinstance(schema, dict):
        raise TypeError(f"Schema must be a dictionary or Pydantic model, not {type(schema)}")

    for k, v in output.items():
        target_type_str = schema.get(k)
        if not target_type_str:
            coerced[k] = v  # No type specified, pass through
            continue
            
        expected_type = type_map.get(target_type_str)
        if not expected_type:
            coerced[k] = v # Unknown type, pass through
            continue

        try:
            coerced[k] = coerce_value(v, expected_type)
        except Exception as e:
            errors[k] = str(e)
            
    if errors:
        raise ValueError(f"Type coercion errors: {errors}")
        
    return coerced 
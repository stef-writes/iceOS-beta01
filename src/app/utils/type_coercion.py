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

def coerce_types(output: dict, schema: dict) -> dict:
    type_map = {"str": str, "int": int, "float": float, "bool": bool}
    coerced = {}
    errors = {}
    for k, v in output.items():
        expected_type = type_map.get(schema.get(k, "str"), str)
        try:
            coerced[k] = coerce_value(v, expected_type)
        except Exception as e:
            errors[k] = str(e)
    if errors:
        raise ValueError(f"Type coercion errors: {errors}")
    return coerced 
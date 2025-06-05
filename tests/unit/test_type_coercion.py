import pytest
from app.utils.type_coercion import coerce_types

def test_int_from_str():
    assert coerce_types({"a": "42"}, {"a": "int"}) == {"a": 42}

def test_float_from_str():
    assert coerce_types({"a": "3.14"}, {"a": "float"}) == {"a": 3.14}

def test_bool_from_str():
    assert coerce_types({"a": "true"}, {"a": "bool"}) == {"a": True}
    assert coerce_types({"a": "no"}, {"a": "bool"}) == {"a": False}

def test_str_from_int():
    assert coerce_types({"a": 42}, {"a": "str"}) == {"a": "42"}

def test_error_on_bad_int():
    with pytest.raises(ValueError):
        coerce_types({"a": "notanumber"}, {"a": "int"})

def test_error_on_bad_float():
    with pytest.raises(ValueError):
        coerce_types({"a": "notafloat"}, {"a": "float"})

def test_bool_from_int():
    assert coerce_types({"a": 1}, {"a": "bool"}) == {"a": True}
    assert coerce_types({"a": 0}, {"a": "bool"}) == {"a": False}

def test_int_from_float():
    assert coerce_types({"a": 42.0}, {"a": "int"}) == {"a": 42} 
import pytest
from app.nodes.error_handling import OpenAIErrorHandler

class DummyAPIError(Exception):
    def __str__(self):
        return "api error"

class DummyRateLimitError(Exception):
    def __str__(self):
        return "rate limit"

class DummyTimeoutError(Exception):
    def __str__(self):
        return "timeout"

class DummyAuthError(Exception):
    def __str__(self):
        return "Invalid API key provided"

class DummyUnknownError(Exception):
    def __str__(self):
        return "something else"

def test_classify_error_apierror():
    assert OpenAIErrorHandler.classify_error(DummyAPIError()) == "UnknownError"

def test_classify_error_ratelimit():
    assert OpenAIErrorHandler.classify_error(DummyRateLimitError()) == "RateLimitError"

def test_classify_error_timeout():
    assert OpenAIErrorHandler.classify_error(DummyTimeoutError()) == "TimeoutError"

def test_classify_error_auth():
    assert OpenAIErrorHandler.classify_error(DummyAuthError()) == "AuthenticationError"

def test_classify_error_unknown():
    assert OpenAIErrorHandler.classify_error(DummyUnknownError()) == "UnknownError"

def test_format_error_message_apierror():
    msg = OpenAIErrorHandler.format_error_message(DummyAPIError())
    assert "api error" in msg.lower()

def test_format_error_message_ratelimit():
    msg = OpenAIErrorHandler.format_error_message(DummyRateLimitError())
    assert "rate limit" in msg.lower()

def test_format_error_message_timeout():
    msg = OpenAIErrorHandler.format_error_message(DummyTimeoutError())
    assert "timed out" in msg.lower()

def test_format_error_message_auth():
    msg = OpenAIErrorHandler.format_error_message(DummyAuthError())
    assert "authentication failed" in msg.lower()

def test_format_error_message_unknown():
    msg = OpenAIErrorHandler.format_error_message(DummyUnknownError())
    assert "something else" in msg.lower() or "unknownerror" in msg.lower() or "unknown error" in msg.lower() 
import pytest
from src.consumers.dlq_processor import categorize_error, ErrorCategory
class TestCategorizeError:
# Test cases for categorize_error function
    def test_timeout_is_transient(self):
        error_message = "Request timeout after 30 seconds"
        result = categorize_error(error_message)
        assert result == ErrorCategory.TRANSIENT

    def test_connection_refused_is_transient(self):
        error_message = "Connection refused by the server"
        result = categorize_error(error_message)
        assert result == ErrorCategory.TRANSIENT

#permament error cases
    def test_invalid_json_is_permanent(self):
        error_message = "Invalid JSON format in the request body"
        result = categorize_error(error_message)
        assert result == ErrorCategory.PERMANENT
    def test_missing_field_is_permanent(self):
        error_message = "Missing required field : amount"
        result = categorize_error(error_message)
        assert result == ErrorCategory.PERMANENT
#unknown error case
    def test_unknown_error(self):
        error_message = "An unexpected error occurred"
        result = categorize_error(error_message)
        assert result == ErrorCategory.UNKNOWN
    def test_empty_string (self):
        error_message = ""
        result = categorize_error(error_message)
        assert result == ErrorCategory.UNKNOWN
#edge cases
    def test_case_insensitive(self):
    # Test that the function is case-insensitive
        error_message = "request TIMEOUT after 30 seconds"
        result = categorize_error(error_message)
        assert result == ErrorCategory.TRANSIENT
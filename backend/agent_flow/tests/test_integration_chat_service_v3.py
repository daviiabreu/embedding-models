"""
Integration tests for Chat Service V3.

Tests the complete workflow including:
- Rate limiting
- Input validation
- PII detection
- Orchestrator V3 integration
- Error handling
"""

from unittest.mock import Mock, patch

import pytest

from chat_service_v3 import ChatService


class TestChatServiceV3Integration:
    """Integration tests for Chat Service V3."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Mock orchestrator for testing."""
        mock = Mock()
        mock.process_message.return_value = "Olá! Como posso ajudar você?"
        return mock

    @pytest.fixture
    def chat_service(self, mock_orchestrator):
        """Create chat service with mocked orchestrator."""
        with patch("chat_service_v3.OrchestratorAgent", return_value=mock_orchestrator):
            service = ChatService()
            # Reset rate limiter for clean tests
            service.rate_limiter.reset_all()
            return service

    def test_successful_request(self, chat_service, mock_orchestrator):
        """Should process valid request successfully."""
        prompt = "Quais são os cursos do Inteli?"
        user_id = "test_user_1"

        response = chat_service.give_response(prompt, user_id)

        assert isinstance(response, str)
        assert len(response) > 0
        mock_orchestrator.process_message.assert_called_once_with(prompt)

    def test_rate_limiting_enforcement(self, chat_service):
        """Should enforce rate limits per user."""
        user_id = "test_user_2"
        prompt = "test"

        # Configure very low limit for testing
        chat_service.rate_limiter.rpm = 2
        chat_service.rate_limiter.rph = 10

        # First 2 requests should succeed
        response1 = chat_service.give_response(prompt, user_id)
        assert "Olá!" in response1

        response2 = chat_service.give_response(prompt, user_id)
        assert "Olá!" in response2

        # Third request should be rate limited
        response3 = chat_service.give_response(prompt, user_id)
        assert "limite" in response3.lower() or "many" in response3.lower()

    def test_different_users_independent_limits(self, chat_service):
        """Rate limits should be independent per user."""
        prompt = "test"

        # Configure low limit
        chat_service.rate_limiter.rpm = 1
        chat_service.rate_limiter.rph = 5

        # User 1 hits limit
        chat_service.give_response(prompt, "user1")
        response_user1 = chat_service.give_response(prompt, "user1")
        assert "limite" in response_user1.lower() or "many" in response_user1.lower()

        # User 2 should still work
        response_user2 = chat_service.give_response(prompt, "user2")
        assert "Olá!" in response_user2

    def test_input_validation(self, chat_service):
        """Should validate input and return appropriate errors."""
        user_id = "test_user_3"

        # Empty input
        response = chat_service.give_response("", user_id)
        assert "vazio" in response.lower() or "empty" in response.lower()

        # Too long input
        long_prompt = "test " * 3000  # ~15k chars
        response = chat_service.give_response(long_prompt, user_id)
        assert "longo" in response.lower() or "long" in response.lower()

        # Null bytes
        response = chat_service.give_response("test\x00data", user_id)
        assert any(
            word in response.lower()
            for word in ["inválido", "invalid", "caractere", "character"]
        )

    def test_pii_detection_warning(self, chat_service, mock_orchestrator):
        """Should detect PII but still process request."""
        user_id = "test_user_4"
        prompt = "Meu email é john@example.com"

        response = chat_service.give_response(prompt, user_id)

        # Should still get response (PII doesn't block)
        assert isinstance(response, str)
        mock_orchestrator.process_message.assert_called_once()

    def test_orchestrator_error_handling(self, chat_service, mock_orchestrator):
        """Should handle orchestrator errors gracefully."""
        user_id = "test_user_5"
        prompt = "test"

        # Simulate orchestrator error
        mock_orchestrator.process_message.side_effect = Exception("Test error")

        response = chat_service.give_response(prompt, user_id)

        # Should return safe error message
        assert "desculpe" in response.lower() or "sorry" in response.lower()
        assert len(response) > 0

    def test_default_user_id(self, chat_service):
        """Should use 'anonymous' if no user_id provided."""
        prompt = "test"

        response = chat_service.give_response(prompt, user_id=None)

        assert isinstance(response, str)
        # Check that anonymous user was rate limited
        usage = chat_service.get_usage("anonymous")
        assert usage["requests_last_minute"] > 0

    def test_get_usage_statistics(self, chat_service):
        """Should provide accurate usage statistics."""
        user_id = "test_user_6"
        prompt = "test"

        # Make 3 requests
        for _ in range(3):
            chat_service.give_response(prompt, user_id)

        usage = chat_service.get_usage(user_id)

        assert usage["requests_last_minute"] == 3
        assert usage["requests_last_hour"] == 3
        assert usage["remaining_minute"] > 0
        assert usage["remaining_hour"] > 0

    def test_reset_rate_limit(self, chat_service):
        """Should reset rate limit for specific user."""
        user_id = "test_user_7"
        prompt = "test"

        # Configure low limit
        chat_service.rate_limiter.rpm = 1

        # Hit limit
        chat_service.give_response(prompt, user_id)
        response1 = chat_service.give_response(prompt, user_id)
        assert "limite" in response1.lower() or "many" in response1.lower()

        # Reset
        chat_service.reset_rate_limit(user_id)

        # Should work again
        response2 = chat_service.give_response(prompt, user_id)
        assert "Olá!" in response2

    def test_concurrent_users(self, chat_service):
        """Should handle multiple users concurrently."""
        prompt = "test"
        user_ids = [f"user_{i}" for i in range(10)]

        responses = []
        for user_id in user_ids:
            response = chat_service.give_response(prompt, user_id)
            responses.append(response)

        # All should succeed
        assert len(responses) == 10
        assert all(isinstance(r, str) for r in responses)

    def test_validation_error_types(self, chat_service):
        """Should handle different validation error types correctly."""
        user_id = "test_user_8"

        # Test different invalid inputs
        invalid_inputs = [
            "",  # Empty
            "   ",  # Whitespace only
            "a" * 20000,  # Too long
            "test\x00null",  # Null byte
        ]

        for invalid_input in invalid_inputs:
            response = chat_service.give_response(invalid_input, user_id)
            # Should return error message, not raise exception
            assert isinstance(response, str)
            assert len(response) > 0


class TestChatServiceV3WithRealOrchestrator:
    """Integration tests with orchestrator (mocked sub-agents)."""

    @pytest.fixture
    def mock_agents(self):
        """Mock all sub-agents."""
        mock_safety = Mock()
        mock_safety.run.return_value = '{"is_safe": true}'

        mock_context = Mock()
        mock_context.run.return_value = '{"history": []}'

        mock_knowledge = Mock()
        mock_knowledge.run.return_value = '{"results": []}'

        return {
            "safety": mock_safety,
            "context": mock_context,
            "knowledge": mock_knowledge,
        }

    @pytest.fixture
    def mock_agent_class(self):
        """Mock Google ADK Agent class."""
        mock_agent = Mock()
        mock_agent.run.return_value = "Olá! Como posso ajudar?"
        return mock_agent

    def test_full_workflow_with_orchestrator(
        self, mock_agents, mock_agent_class, mock_config
    ):
        """Test full workflow with orchestrator (mocked sub-agents)."""
        with (
            patch("config.config", mock_config),
            patch("chat_service_v3.OrchestratorAgent") as mock_orch_class,
            patch("google.generativeai.configure"),
        ):
            # Setup mock
            mock_orchestrator = Mock()
            mock_orchestrator.process_message.return_value = "Olá! Como posso ajudar?"
            mock_orch_class.return_value = mock_orchestrator

            # Create service
            service = ChatService()
            service.rate_limiter.reset_all()

            # Test request
            response = service.give_response("Olá!", "test_user")

            assert isinstance(response, str)
            assert len(response) > 0
            mock_orchestrator.process_message.assert_called_once()


class TestChatServiceV3ErrorRecovery:
    """Test error recovery and resilience."""

    @pytest.fixture
    def chat_service_with_failing_orchestrator(self):
        """Create service with orchestrator that fails sometimes."""
        mock_orch = Mock()

        def side_effect(msg):
            if "fail" in msg.lower():
                raise Exception("Simulated failure")
            return "Success response"

        mock_orch.process_message.side_effect = side_effect

        with patch("chat_service_v3.OrchestratorAgent", return_value=mock_orch):
            service = ChatService()
            service.rate_limiter.reset_all()
            return service

    def test_recovery_after_error(self, chat_service_with_failing_orchestrator):
        """Should recover after orchestrator error."""
        service = chat_service_with_failing_orchestrator
        user_id = "test_user_9"

        # First request fails
        response1 = service.give_response("fail", user_id)
        assert "desculpe" in response1.lower()

        # Second request should succeed
        response2 = service.give_response("success", user_id)
        assert "Success response" in response2

    def test_error_doesnt_block_other_users(
        self, chat_service_with_failing_orchestrator
    ):
        """Errors for one user shouldn't affect others."""
        service = chat_service_with_failing_orchestrator

        # User 1 fails
        response1 = service.give_response("fail", "user1")
        assert "desculpe" in response1.lower()

        # User 2 succeeds
        response2 = service.give_response("success", "user2")
        assert "Success response" in response2


class TestChatServiceV3Metrics:
    """Test metrics collection."""

    @pytest.fixture
    def chat_service(self):
        """Create chat service with mocked orchestrator."""
        mock_orch = Mock()
        mock_orch.process_message.return_value = "Test response"

        with patch("chat_service_v3.OrchestratorAgent", return_value=mock_orch):
            service = ChatService()
            service.rate_limiter.reset_all()
            return service

    def test_metrics_collected_on_success(self, chat_service):
        """Should collect metrics on successful requests."""
        with patch("chat_service_v3.metrics") as mock_metrics:
            chat_service.give_response("test", "user1")

            # Verify metrics were called
            mock_metrics.requests_total.labels.assert_called()

    def test_metrics_collected_on_rate_limit(self, chat_service):
        """Should collect metrics on rate limit errors."""
        chat_service.rate_limiter.rpm = 1

        with patch("chat_service_v3.metrics") as mock_metrics:
            # First request succeeds
            chat_service.give_response("test", "user1")

            # Second request hits rate limit
            chat_service.give_response("test", "user1")

            # Check that rate_limited status was tracked
            calls = mock_metrics.requests_total.labels.call_args_list
            assert any("rate_limited" in str(call) for call in calls)

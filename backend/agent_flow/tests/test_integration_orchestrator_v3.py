"""
Integration tests for Orchestrator Agent V3.

Tests the orchestrator workflow without personality agent:
- Tool integration (retrieve_inteli_knowledge, check_content_safety, etc.)
- Error handling and recovery
- Conversation history management

V3.1 Changes:
- Orchestrator now uses tools directly instead of sub_agents
- Tests updated to verify tools are properly configured
"""

from unittest.mock import Mock, patch

import pytest

from agents.orchestrator_agent_v3 import OrchestratorAgent, create_orchestrator_agent


class TestOrchestratorV3Creation:
    """Test orchestrator V3 creation and initialization."""

    def test_create_with_model(self, mock_config):
        """Should create orchestrator with specified model."""
        with (
            patch("config.config", mock_config),
            patch("agents.orchestrator_agent_v3.Agent") as mock_agent_class,
        ):
            orchestrator = create_orchestrator_agent(model="gemini-2.5-pro")

            # Verify agent was created with correct parameters
            mock_agent_class.assert_called_once()
            call_kwargs = mock_agent_class.call_args[1]

            assert call_kwargs["name"] == "orchestrator_agent_v3"
            assert call_kwargs["model"] == "gemini-2.5-pro"

    def test_create_with_tools(self, mock_config):
        """Should create orchestrator with tools (not sub_agents)."""
        with (
            patch("config.config", mock_config),
            patch("agents.orchestrator_agent_v3.Agent") as mock_agent_class,
        ):
            orchestrator = create_orchestrator_agent()

            # Verify agent was created with tools
            mock_agent_class.assert_called_once()
            call_kwargs = mock_agent_class.call_args[1]

            # Should have tools, not sub_agents
            assert "tools" in call_kwargs
            assert len(call_kwargs["tools"]) == 4  # 4 tools

    def test_no_sub_agents(self, mock_config):
        """Should NOT use sub_agents (uses tools instead)."""
        with (
            patch("config.config", mock_config),
            patch("agents.orchestrator_agent_v3.Agent") as mock_agent_class,
        ):
            orchestrator = create_orchestrator_agent()

            # Get call kwargs
            call_kwargs = mock_agent_class.call_args[1]

            # Should NOT have sub_agents
            assert (
                "sub_agents" not in call_kwargs or call_kwargs.get("sub_agents") is None
            )


class TestOrchestratorV3Processing:
    """Test message processing with orchestrator V3."""

    @pytest.fixture
    def mock_agent(self):
        """Mock Google ADK Agent."""
        mock = Mock()
        mock.run.return_value = "Olá! Como posso ajudar você?"
        mock.name = "orchestrator_agent_v3"
        return mock

    @pytest.fixture
    def orchestrator(self, mock_agent, mock_config):
        """Create orchestrator with mocked agent."""
        with (
            patch("config.config", mock_config),
            patch(
                "agents.orchestrator_agent_v3.create_orchestrator_agent",
                return_value=mock_agent,
            ),
            patch("google.generativeai.configure"),
        ):
            orch = OrchestratorAgent(model="gemini-2.5-pro")
            return orch

    def test_process_simple_message(self, orchestrator, mock_agent):
        """Should process simple message successfully."""
        message = "Olá!"
        response = orchestrator.process_message(message)

        assert isinstance(response, str)
        assert len(response) > 0
        mock_agent.run.assert_called_once_with(message)

    def test_process_question_about_inteli(self, orchestrator, mock_agent):
        """Should process questions about Inteli."""
        mock_agent.run.return_value = "O Inteli oferece cursos de tecnologia."

        message = "Quais são os cursos do Inteli?"
        response = orchestrator.process_message(message)

        assert "Inteli" in response or "cursos" in response
        mock_agent.run.assert_called_once()

    def test_conversation_history_tracking(self, orchestrator, mock_agent):
        """Should track conversation history locally."""
        # First message
        orchestrator.process_message("Olá!")
        assert len(orchestrator.conversation_history) == 2  # user + assistant

        # Second message
        orchestrator.process_message("Como vai?")
        assert len(orchestrator.conversation_history) == 4  # 2 exchanges

        # Verify structure
        assert orchestrator.conversation_history[0]["role"] == "user"
        assert orchestrator.conversation_history[1]["role"] == "assistant"

    def test_get_conversation_history(self, orchestrator):
        """Should retrieve conversation history."""
        orchestrator.process_message("Message 1")
        orchestrator.process_message("Message 2")

        history = orchestrator.get_conversation_history()

        assert len(history) == 4
        assert all("role" in msg and "content" in msg for msg in history)

    def test_clear_conversation_history(self, orchestrator):
        """Should clear conversation history."""
        orchestrator.process_message("Message 1")
        orchestrator.process_message("Message 2")

        assert len(orchestrator.conversation_history) > 0

        orchestrator.clear_history()

        assert len(orchestrator.conversation_history) == 0


class TestOrchestratorV3ErrorHandling:
    """Test error handling in orchestrator V3."""

    @pytest.fixture
    def orchestrator_with_failures(self, mock_config):
        """Create orchestrator that simulates various failures."""
        mock_agent = Mock()

        def side_effect(message):
            if "validation_error" in message:
                raise ValueError("Validation failed")
            elif "key_error" in message:
                raise KeyError("Missing key")
            elif "unknown_error" in message:
                raise RuntimeError("Unknown error")
            return "Success response"

        mock_agent.run.side_effect = side_effect

        with (
            patch("config.config", mock_config),
            patch(
                "agents.orchestrator_agent_v3.create_orchestrator_agent",
                return_value=mock_agent,
            ),
            patch("google.generativeai.configure"),
        ):
            return OrchestratorAgent()

    def test_validation_error_handling(self, orchestrator_with_failures):
        """Should handle validation errors gracefully."""
        response = orchestrator_with_failures.process_message("validation_error test")

        assert isinstance(response, str)
        assert "desculpe" in response.lower()
        assert "processar" in response.lower()

    def test_key_error_handling(self, orchestrator_with_failures):
        """Should handle missing data errors."""
        response = orchestrator_with_failures.process_message("key_error test")

        assert isinstance(response, str)
        assert "desculpe" in response.lower()
        assert "problema" in response.lower()

    def test_unknown_error_handling(self, orchestrator_with_failures):
        """Should handle unexpected errors gracefully."""
        response = orchestrator_with_failures.process_message("unknown_error test")

        assert isinstance(response, str)
        assert "desculpe" in response.lower()
        assert "probleminha" in response.lower() or "técnico" in response.lower()

    def test_error_recovery(self, orchestrator_with_failures):
        """Should recover after errors."""
        # Error
        response1 = orchestrator_with_failures.process_message("validation_error")
        assert "desculpe" in response1.lower()

        # Recovery
        response2 = orchestrator_with_failures.process_message("normal message")
        assert "Success response" in response2


class TestOrchestratorV3Workflow:
    """Test complete orchestrator workflow (4 stages)."""

    @pytest.fixture
    def orchestrator_with_workflow_tracking(self, mock_config):
        """Create orchestrator that tracks agent calls."""
        mock_agent = Mock()

        # Track which stages were called
        workflow_calls = []

        def track_workflow(message):
            workflow_calls.append(message)
            return f"Processed: {message}"

        mock_agent.run.side_effect = track_workflow
        mock_agent.workflow_calls = workflow_calls

        with (
            patch("config.config", mock_config),
            patch(
                "agents.orchestrator_agent_v3.create_orchestrator_agent",
                return_value=mock_agent,
            ),
            patch("google.generativeai.configure"),
        ):
            return OrchestratorAgent(), mock_agent

    def test_workflow_stages_executed(self, orchestrator_with_workflow_tracking):
        """Should execute workflow stages in order."""
        orchestrator, mock_agent = orchestrator_with_workflow_tracking

        message = "Test message"
        response = orchestrator.process_message(message)

        # Agent should be called
        assert len(mock_agent.workflow_calls) == 1
        assert mock_agent.workflow_calls[0] == message

    def test_multiple_messages_workflow(self, orchestrator_with_workflow_tracking):
        """Should handle multiple messages correctly."""
        orchestrator, mock_agent = orchestrator_with_workflow_tracking

        messages = ["Message 1", "Message 2", "Message 3"]

        for msg in messages:
            response = orchestrator.process_message(msg)
            assert f"Processed: {msg}" in response

        # All messages should be tracked
        assert len(mock_agent.workflow_calls) == 3


class TestOrchestratorV3Integration:
    """Full integration tests with tools."""

    @pytest.fixture
    def full_integration_setup(self, mock_config):
        """Setup orchestrator with mocked tools that simulate real behavior."""
        # Mock main agent
        mock_orchestrator_agent = Mock()
        mock_orchestrator_agent.run.return_value = (
            "Ola! O Inteli oferece cursos de tecnologia."
        )
        mock_orchestrator_agent.name = "orchestrator_agent_v3"

        with (
            patch("config.config", mock_config),
            patch(
                "agents.orchestrator_agent_v3.Agent",
                return_value=mock_orchestrator_agent,
            ),
            patch("google.generativeai.configure"),
        ):
            orch = OrchestratorAgent()

            return {
                "orchestrator": orch,
                "main_agent": mock_orchestrator_agent,
            }

    def test_full_workflow_integration(self, full_integration_setup):
        """Test complete workflow with all components."""
        setup = full_integration_setup
        orchestrator = setup["orchestrator"]
        mock_agent = setup["main_agent"]

        # Process message
        response = orchestrator.process_message("Quais sao os cursos?")

        # Should get response
        assert isinstance(response, str)
        assert len(response) > 0

        # Main agent should be called
        mock_agent.run.assert_called_once()

    def test_conversation_flow(self, full_integration_setup):
        """Test multi-turn conversation."""
        orchestrator = full_integration_setup["orchestrator"]

        # Turn 1
        response1 = orchestrator.process_message("Ola!")
        assert len(response1) > 0

        # Turn 2
        response2 = orchestrator.process_message("Quais cursos voces oferecem?")
        assert len(response2) > 0

        # Turn 3
        response3 = orchestrator.process_message("Me conte mais sobre admissao")
        assert len(response3) > 0

        # Should have 6 messages in history (3 turns)
        assert len(orchestrator.conversation_history) == 6


class TestOrchestratorV3Configuration:
    """Test configuration and initialization."""

    def test_requires_google_api_key(self, mock_config):
        """Should require GOOGLE_API_KEY in config."""
        # Remove API key
        mock_config.model.GOOGLE_API_KEY = None

        with patch("config.config", mock_config), pytest.raises(ValueError):
            OrchestratorAgent()

    def test_uses_configured_model(self, mock_config):
        """Should use model from config or parameter."""
        mock_agent = Mock()

        with (
            patch("config.config", mock_config),
            patch(
                "agents.orchestrator_agent_v3.create_orchestrator_agent",
                return_value=mock_agent,
            ) as mock_create,
            patch("google.generativeai.configure"),
        ):
            orch = OrchestratorAgent(model="custom-model")

            # Should use custom model
            mock_create.assert_called_once()
            assert mock_create.call_args[1]["model"] == "custom-model"

    def test_api_configuration(self, mock_config):
        """Should configure Google GenerativeAI with API key."""
        mock_agent = Mock()

        with (
            patch("config.config", mock_config),
            patch(
                "agents.orchestrator_agent_v3.create_orchestrator_agent",
                return_value=mock_agent,
            ),
            patch("google.generativeai.configure") as mock_configure,
        ):
            orch = OrchestratorAgent()

            # Should configure API
            mock_configure.assert_called_once_with(
                api_key=mock_config.model.GOOGLE_API_KEY
            )


class TestOrchestratorV3Performance:
    """Test performance characteristics."""

    @pytest.fixture
    def orchestrator_with_timing(self, mock_config):
        """Create orchestrator that tracks timing."""
        mock_agent = Mock()
        mock_agent.run.return_value = "Response"

        with (
            patch("config.config", mock_config),
            patch(
                "agents.orchestrator_agent_v3.create_orchestrator_agent",
                return_value=mock_agent,
            ),
            patch("google.generativeai.configure"),
        ):
            return OrchestratorAgent(), mock_agent

    def test_single_agent_call_per_message(self, orchestrator_with_timing):
        """Should make single orchestrator agent call per message."""
        orchestrator, mock_agent = orchestrator_with_timing

        orchestrator.process_message("Test message")

        # Should call agent exactly once
        assert mock_agent.run.call_count == 1

    def test_no_redundant_calls(self, orchestrator_with_timing):
        """Should not make redundant agent calls."""
        orchestrator, mock_agent = orchestrator_with_timing

        # Process 3 messages
        for i in range(3):
            orchestrator.process_message(f"Message {i}")

        # Should be exactly 3 calls (one per message)
        assert mock_agent.run.call_count == 3

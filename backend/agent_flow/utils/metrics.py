"""Prometheus metrics for monitoring."""

from prometheus_client import Counter, Gauge, Histogram, Info

# Request metrics
requests_total = Counter(
    "agent_flow_requests_total", "Total number of requests", ["agent", "status"]
)

request_latency = Histogram(
    "agent_flow_request_latency_seconds",
    "Request latency in seconds",
    ["agent"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# LLM metrics
llm_calls_total = Counter(
    "agent_flow_llm_calls_total", "Total LLM API calls", ["model", "agent"]
)

llm_tokens_total = Counter(
    "agent_flow_llm_tokens_total",
    "Total tokens consumed",
    ["model", "type"],  # type: prompt, completion
)

# Safety metrics
safety_blocks_total = Counter(
    "agent_flow_safety_blocks_total",
    "Total safety blocks",
    ["reason"],  # reason: pii, jailbreak, toxicity, off_topic
)

# RAG metrics
rag_queries_total = Counter("agent_flow_rag_queries_total", "Total RAG queries")

rag_chunks_retrieved = Histogram(
    "agent_flow_rag_chunks_retrieved",
    "Number of chunks retrieved per query",
    buckets=[0, 1, 5, 10, 20, 50, 100, 300],
)

# Error metrics
errors_total = Counter("agent_flow_errors_total", "Total errors", ["type", "agent"])

# System info
system_info = Info("agent_flow_system", "System information")

# Active sessions
active_sessions = Gauge("agent_flow_active_sessions", "Number of active user sessions")

# Cache metrics
cache_hits_total = Counter(
    "agent_flow_cache_hits_total",
    "Total cache hits",
    ["cache_type"],  # cache_type: embedding, rag, etc
)

cache_misses_total = Counter(
    "agent_flow_cache_misses_total", "Total cache misses", ["cache_type"]
)

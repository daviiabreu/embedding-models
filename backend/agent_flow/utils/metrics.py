"""Prometheus metrics for monitoring."""

from prometheus_client import Counter, Gauge, Histogram, Info, REGISTRY

# Check if metrics are already registered to avoid duplicates
def get_or_create_counter(name, doc, labels):
    """Get existing counter or create new one."""
    try:
        return REGISTRY._names_to_collectors.get(name)
    except:
        pass
    return Counter(name, doc, labels)

def get_or_create_histogram(name, doc, labels, buckets):
    """Get existing histogram or create new one."""
    try:
        return REGISTRY._names_to_collectors.get(name)
    except:
        pass
    return Histogram(name, doc, labels, buckets=buckets)

def get_or_create_gauge(name, doc, labels):
    """Get existing gauge or create new one."""
    try:
        return REGISTRY._names_to_collectors.get(name)
    except:
        pass
    return Gauge(name, doc, labels)

# Request metrics
requests_total = get_or_create_counter(
    "agent_flow_requests_total", "Total number of requests", ["agent", "status"]
)

request_latency = get_or_create_histogram(
    "agent_flow_request_latency_seconds",
    "Request latency in seconds",
    ["agent"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# LLM metrics
llm_calls_total = get_or_create_counter(
    "agent_flow_llm_calls_total", "Total LLM API calls", ["model", "agent"]
)

llm_tokens_total = get_or_create_counter(
    "agent_flow_llm_tokens_total",
    "Total tokens consumed",
    ["model", "type"],  # type: prompt, completion
)

# Safety metrics
safety_blocks_total = get_or_create_counter(
    "agent_flow_safety_blocks_total",
    "Total safety blocks",
    ["reason"],  # reason: pii, jailbreak, toxicity, off_topic
)

# RAG metrics
rag_queries_total = get_or_create_counter("agent_flow_rag_queries_total", "Total RAG queries", [])

rag_chunks_retrieved = get_or_create_histogram(
    "agent_flow_rag_chunks_retrieved",
    "Number of chunks retrieved per query",
    [],
    buckets=[0, 1, 5, 10, 20, 50, 100, 300],
)

# Error metrics
errors_total = get_or_create_counter("agent_flow_errors_total", "Total errors", ["type", "agent"])

# System info
try:
    system_info = REGISTRY._names_to_collectors.get("agent_flow_system")
    if not system_info:
        system_info = Info("agent_flow_system", "System information")
except:
    system_info = Info("agent_flow_system", "System information")

# Active sessions
active_sessions = get_or_create_gauge("agent_flow_active_sessions", "Number of active user sessions", [])

# Cache metrics
cache_hits_total = get_or_create_counter(
    "agent_flow_cache_hits_total",
    "Total cache hits",
    ["cache_type"],  # cache_type: embedding, rag, etc
)

cache_misses_total = get_or_create_counter(
    "agent_flow_cache_misses_total", "Total cache misses", ["cache_type"]
)

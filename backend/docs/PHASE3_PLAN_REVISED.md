# Phase 3: Code Quality & Selective Optimization

**Status**: 📋 **READY TO START**
**Date**: December 10, 2025
**Estimated Duration**: 1-2 days
**Priority**: HIGH

---

## TL;DR - Phase 3 Goals

🎯 **Centralize all constants and magic numbers**
⚡ **Add async where it provides real benefit**
🧹 **Clean up dead code and legacy files**

---

## Phase 3 Tasks

### Task 3.2: Centralize Constants (2-3 hours)

**Goal**: Move all hardcoded values to centralized configuration
**Impact**: Better maintainability, easier tuning, no magic numbers

#### Current Issues

- Magic numbers scattered throughout code
- Hardcoded thresholds in multiple files
- Configuration values duplicated
- Difficult to tune system parameters

#### Changes Needed

**1. Create centralized constants file** (`utils/constants.py`):

```python
"""Centralized constants for agent flow."""

# Heuristics thresholds
FORMALITY_THRESHOLD = 0.6
VERBOSITY_SHORT_WORDS = 10
VERBOSITY_LONG_WORDS = 30
ENGAGEMENT_HIGH_THRESHOLD = 0.7
ENGAGEMENT_LOW_THRESHOLD = 0.3

# PII Detection
CPF_LENGTH = 11
CNPJ_LENGTH = 14
CEP_PATTERN_LENGTH = 8

# Rate Limiting (can override from config)
DEFAULT_REQUESTS_PER_MINUTE = 60
DEFAULT_REQUESTS_PER_HOUR = 500

# Validation
DEFAULT_MAX_INPUT_LENGTH = 10_000
MIN_INPUT_LENGTH = 1

# Agent Configuration
MAX_RETRY_ATTEMPTS = 3
AGENT_TIMEOUT_SECONDS = 30

# Logging
LOG_MAX_MESSAGE_LENGTH = 500

# Cache (for future use)
DEFAULT_CACHE_TTL_SECONDS = 3600
```

**2. Extract constants from existing files**:

- `utils/heuristics.py` - Thresholds for detection
- `utils/validation.py` - Length limits
- `utils/pii_detector.py` - Pattern lengths
- `utils/rate_limiter.py` - Default limits
- `agents/orchestrator_agent_v3.py` - Timeouts, retries

**3. Update all files to use centralized constants**

#### Files to Modify

- [ ] Create `utils/constants.py`
- [ ] Update `utils/heuristics.py` - Replace hardcoded thresholds
- [ ] Update `utils/validation.py` - Replace length constants
- [ ] Update `utils/pii_detector.py` - Replace pattern constants
- [ ] Update `utils/rate_limiter.py` - Use constants as defaults
- [ ] Update `agents/orchestrator_agent_v3.py` - Use timeout constants
- [ ] Update `utils/__init__.py` - Export constants

#### Acceptance Criteria

- ✅ All magic numbers replaced with named constants
- ✅ Constants grouped logically
- ✅ Constants documented with comments
- ✅ All tests still passing
- ✅ Easy to tune system parameters from one place

---

### Task 3.4: Add Async Where Beneficial (4-6 hours)

**Goal**: Add async operations where they provide real performance benefit
**Impact**: Faster parallel operations, better concurrency for I/O-bound tasks

#### Strategy: Selective Async (Not Full Refactoring)

Only add async where it **measurably improves performance**:

1. ✅ **Parallel agent calls** (safety + context in parallel)
2. ✅ **Async Qdrant queries** (if SDK supports)
3. ✅ **Async LLM calls** (if Google SDK supports)
4. ❌ **NOT converting entire codebase** (too much effort, minimal gain)

#### Changes Needed

**1. Add async utilities** (`utils/async_helpers.py`):

```python
"""Async helper functions."""
import asyncio
from typing import Any, Callable, List

async def run_parallel(*tasks) -> List[Any]:
    """Run multiple async tasks in parallel."""
    return await asyncio.gather(*tasks, return_exceptions=True)

async def run_with_timeout(coro, timeout: float):
    """Run coroutine with timeout."""
    return await asyncio.wait_for(coro, timeout=timeout)

def async_wrap(func: Callable) -> Callable:
    """Wrap sync function to run in executor."""
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    return wrapper
```

**2. Add parallel agent execution** (orchestrator_agent_v3.py):

```python
async def process_message_parallel(self, user_message: str) -> str:
    """
    Process message with parallel agent calls where beneficial.

    Workflow:
    1. Safety check (required first)
    2. Context + Knowledge in PARALLEL (both are I/O bound)
    3. Generate response
    """
    # Step 1: Safety check (must be first)
    is_safe = await self._check_safety_async(user_message)
    if not is_safe:
        return "Desculpe, não posso processar essa mensagem."

    # Step 2: Parallel retrieval (context + knowledge)
    context_task = self._get_context_async(user_message)
    knowledge_task = self._get_knowledge_async(user_message)

    context, knowledge = await asyncio.gather(
        context_task,
        knowledge_task,
        return_exceptions=True
    )

    # Step 3: Generate response with context + knowledge
    response = await self._generate_response_async(
        user_message, context, knowledge
    )

    return response
```

**3. Add async Qdrant queries** (tools/knowledge_tools.py):

```python
async def retrieve_inteli_knowledge_async(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.7
) -> dict:
    """Async version of knowledge retrieval."""
    # If qdrant-client supports async
    results = await qdrant_client.search_async(
        collection_name=config.rag.QDRANT_COLLECTION,
        query_vector=embedding,
        limit=top_k,
        score_threshold=score_threshold
    )
    return format_results(results)
```

**4. Keep sync version as default** (backward compatibility):

```python
# chat_service_v3.py keeps sync interface
def give_response(self, prompt: str, user_id: str | None = None) -> str:
    """Sync version - uses async internally if available."""
    if hasattr(self.orchestrator, 'process_message_parallel'):
        # Run async version in event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in async context
            return await self.orchestrator.process_message_parallel(prompt)
        else:
            # Create new event loop
            return loop.run_until_complete(
                self.orchestrator.process_message_parallel(prompt)
            )
    else:
        # Fall back to sync version
        return self.orchestrator.process_message(prompt)
```

#### Files to Modify

- [ ] Create `utils/async_helpers.py`
- [ ] Update `agents/orchestrator_agent_v3.py` - Add parallel execution
- [ ] Update `tools/knowledge_tools.py` - Add async Qdrant (if supported)
- [ ] Update `tools/context_tools.py` - Add async context retrieval
- [ ] Keep `chat_service_v3.py` sync (with async internal calls)
- [ ] Add async tests to existing test files

#### Acceptance Criteria

- ✅ Parallel agent calls working (context + knowledge)
- ✅ Measurable performance improvement (20-30% faster)
- ✅ Backward compatibility maintained (sync interface)
- ✅ All tests passing (sync and async)
- ✅ Error handling for parallel operations
- ✅ No breaking changes to API

---

### Task 3.7: Clean Up Dead Code (1-2 hours)

**Goal**: Remove legacy files and clean up codebase
**Impact**: Easier maintenance, reduced confusion, cleaner repository

#### Dead Code to Remove

**1. Old orchestrator versions**:

- [ ] `agents/orchestrator_agent.py` (original version)
- [ ] `agents/orchestrator_agent_v1_backup.py` (backup)
- [ ] Any other `*_backup.py` files

**2. Unused agent files**:

- [ ] `agents/personality_agent.py` (removed in Phase 2)
- [ ] `agents/tour_agent.py` (if not used)

**3. Old tool files**:

- [ ] `tools/personality_tools.py` (if personality agent removed)
- [ ] Any deprecated tool files

**4. Temporary/test files**:

- [ ] Any `test_*.py.bak` files
- [ ] `__pycache__` directories (add to .gitignore if not there)
- [ ] Any `.pyc` files

**5. Clean up imports**:

- [ ] Remove unused imports across all files
- [ ] Fix import order (standard → third-party → local)
- [ ] Remove circular imports if any

#### Files to Check for Dead Code

```bash
# Find backup files
find . -name "*_backup.py"
find . -name "*.bak"

# Find potentially unused imports
# (manual review needed)
rg "^import|^from" --no-heading | sort | uniq

# Find large commented blocks
rg "^#.*\n#.*\n#.*\n#.*" -A 10
```

#### Process

1. **Identify dead code**:

   - Review each file in `agents/` and `tools/`
   - Check git history for last use
   - Verify no active imports

2. **Create backup branch**:

   ```bash
   git checkout -b backup/pre-phase3-cleanup
   git push origin backup/pre-phase3-cleanup
   ```

3. **Remove dead code**:

   - Delete files
   - Update imports
   - Remove from `__init__.py` exports

4. **Clean imports**:

   ```bash
   # Use autoflake to remove unused imports
   autoflake --in-place --remove-all-unused-imports **/*.py

   # Use isort to organize imports
   isort **/*.py
   ```

5. **Verify tests still pass**:
   ```bash
   pytest tests/ -v
   ```

#### Files to Remove (Confirmed Dead Code)

- [ ] `agents/orchestrator_agent_v1_backup.py` - Backup file
- [ ] `agents/personality_agent.py` - Removed in Phase 2
- [ ] `tools/personality_tools.py` - Removed in Phase 2
- [ ] Any files not imported anywhere

#### Files to Update

- [ ] `agents/__init__.py` - Remove dead imports
- [ ] `tools/__init__.py` - Remove dead imports
- [ ] All files with unused imports

#### Acceptance Criteria

- ✅ All backup files removed
- ✅ All unused agent/tool files removed
- ✅ No unused imports remaining
- ✅ All tests still passing
- ✅ Clean `git status` (no untracked files)
- ✅ Backup branch created for safety

---

## Implementation Plan

### Timeline: 1-2 Days

#### Day 1 Morning: Task 3.7 (Clean Up)

**Duration**: 1-2 hours
**Rationale**: Start with cleanup to reduce noise

1. Create backup branch
2. Identify and remove dead code
3. Clean up imports
4. Run tests to verify nothing broke

#### Day 1 Afternoon: Task 3.2 (Constants)

**Duration**: 2-3 hours
**Rationale**: Foundation for better maintainability

1. Create `utils/constants.py`
2. Extract constants from all files
3. Update files to use centralized constants
4. Run tests to verify behavior unchanged

#### Day 2: Task 3.4 (Async)

**Duration**: 4-6 hours
**Rationale**: Most complex task, needs full day

1. Create async utilities
2. Add parallel agent execution
3. Add async Qdrant queries (if supported)
4. Test performance improvements
5. Ensure backward compatibility

---

## Testing Strategy

### For Each Task

**Task 3.2 (Constants)**:

- ✅ All existing tests pass
- ✅ Verify constants are used correctly
- ✅ Test that changing constants affects behavior

**Task 3.4 (Async)**:

- ✅ Add async-specific tests
- ✅ Test parallel execution
- ✅ Test error handling in parallel operations
- ✅ Benchmark performance improvement
- ✅ Test backward compatibility (sync still works)

**Task 3.7 (Cleanup)**:

- ✅ All tests pass after cleanup
- ✅ No import errors
- ✅ Git history preserved (backup branch)

---

## Success Metrics

### Completion Criteria

| Task              | Success Metric           | Target        |
| ----------------- | ------------------------ | ------------- |
| **3.2 Constants** | No magic numbers in code | 100%          |
| **3.2 Constants** | All tests passing        | ✅            |
| **3.4 Async**     | Performance improvement  | 20-30% faster |
| **3.4 Async**     | Backward compatible      | ✅            |
| **3.7 Cleanup**   | Lines of code removed    | 1000+         |
| **3.7 Cleanup**   | All tests passing        | ✅            |

### Performance Targets

| Metric                  | Before        | After     | Improvement   |
| ----------------------- | ------------- | --------- | ------------- |
| **Parallel Operations** | Sequential    | Parallel  | 20-30% faster |
| **Code Cleanliness**    | Legacy files  | Clean     | -1000+ LOC    |
| **Maintainability**     | Magic numbers | Constants | Easy to tune  |

---

## Risk Assessment

### Low Risk 🟢

1. **Task 3.7 (Cleanup)**

   - Risk: Accidentally remove needed code
   - Mitigation: Backup branch, careful review
   - Impact: Low (easy to restore)

2. **Task 3.2 (Constants)**
   - Risk: Missing a constant somewhere
   - Mitigation: Comprehensive testing
   - Impact: Low (easy to fix)

### Medium Risk 🟡

3. **Task 3.4 (Async)**
   - Risk: Breaking changes to API
   - Mitigation: Maintain sync interface, gradual adoption
   - Impact: Medium (need thorough testing)

---

## Rollback Plan

If any issues arise:

1. **Task 3.7**: Restore from backup branch
2. **Task 3.2**: Revert constants commit
3. **Task 3.4**: Disable async via feature flag, use sync version

All tasks are designed to be **independently reversible**.

---

## Phase 3 Deliverables

### Code Changes

- [ ] `utils/constants.py` - New centralized constants file
- [ ] `utils/async_helpers.py` - New async utilities
- [ ] All files updated to use constants
- [ ] Async parallel execution in orchestrator
- [ ] Dead code removed

### Documentation

- [ ] Update README with constants usage
- [ ] Document async performance improvements
- [ ] Update architecture docs (removed components)

### Tests

- [ ] All existing tests passing (117 tests)
- [ ] New async tests added
- [ ] Performance benchmarks documented

---

## Post-Phase 3

After completing these tasks:

1. **System will be**:

   - ✅ Cleaner and more maintainable
   - ✅ Faster (20-30% improvement in parallel operations)
   - ✅ Easier to tune (centralized constants)

2. **Next steps** (optional future phases):
   - Caching layer (if needed for cost reduction)
   - Redis for distributed rate limiting (if multi-instance needed)
   - Advanced monitoring (if observability needed)

---

## Conclusion

Phase 3 focuses on **pragmatic improvements** that provide **immediate value**:

- 🎯 **Better maintainability** (centralized constants)
- ⚡ **Better performance** (selective async, 20-30% faster)
- 🧹 **Cleaner codebase** (remove 1000+ lines of dead code)

**Estimated Effort**: 7-11 hours (1-2 days)
**Risk Level**: LOW (all changes reversible)
**Impact**: HIGH (better code quality, faster performance)

---

**Document Version**: 1.0 (Revised per requirements)
**Created**: December 10, 2025
**Status**: 📋 **READY TO START**

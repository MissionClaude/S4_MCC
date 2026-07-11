"""Integration tests — end-to-end agent workflows with a mocked API.

These tests wire the real agent, loop, tools, and context components
together. The Anthropic SDK is replaced with a scripted fake, but
everything else is the real production code.

The four test files in this package cover the spec's key flows:

* ``test_full_workflow.py`` — coordinator dispatches subagents,
  GenerateAgent writes a file, error handling across the boundary.
* ``test_pipeline.py`` — non-interactive pipeline mode (json output,
  session isolation, multi-pass analysis).
* ``test_context.py`` — scratchpad survives iterations, /compact
  reduces message count, PostToolUse hooks trim large outputs.
* ``test_error_scenarios.py`` — rate limit retry, auth failure, tool
  failure feeding back to the model.

Shared fixtures live in ``conftest.py`` at the package root.
"""

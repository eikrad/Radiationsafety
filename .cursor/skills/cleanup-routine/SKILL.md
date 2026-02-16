---
name: cleanup-routine
description: Runs a pre-commit cleanup workflow: review recent code, remove dead or redundant code, simplify logic and control flow, consider performance, run backend and frontend tests, suggest reusable patterns. Use before committing changes or when the user asks for codebase cleanup.
---

# Cleanup routine

Keep the codebase **fast and clean**. Before committing changes, run this workflow.

## Workflow

1. **Review recent code** and look for opportunities.

2. **Apply cleanup by:**
   - Looking for redundant or dead code.
   - Straightening logic and control flow.
   - Simplifying excessive or repetitive code.
   - Is the code in the correct part/file
   - Considering performance (unnecessary work, parallelization, caching).

3. **Run the test suite.** Fix any failures before finishing.
   - Backend: `uv run pytest tests/ -v`
   - Frontend: `cd frontend && npm run test -- --run`

4. **Identify reusable patterns and optional abstractions;** suggest them briefly to the user and keep suggestions non-blocking.

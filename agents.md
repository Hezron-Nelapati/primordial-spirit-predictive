# agents.md

## Purpose
This file defines how coding agents must implement the SaaS product in a reliable, context-aware, reviewable manner.

---

## Core Operating Rules

### 1. Respect Context Window
- Assume a maximum working context window of **200K tokens**.
- Each implementation phase must fit safely within this limit.
- Do not attempt large multi-module implementation in one pass if it risks exceeding context clarity.
- Split work into smaller phases when needed.

### 2. Work in Phases
Each implementation must be divided into clearly scoped phases.

Each phase must contain:
- Phase name
- Goal
- In scope
- Out of scope
- Dependencies
- Files/modules expected to change
- Expected output

### 3. Keep Changes Reviewable
- Prefer small, reviewable, PR-sized changes.
- Avoid touching unrelated files.
- Avoid broad refactors unless the phase explicitly allows them.
- If a task is too large, first split it into sub-phases.

### 4. Architecture First, Then Code
Before implementing, identify:
- affected modules
- relevant entities/models
- API or interface contracts involved
- DB changes required
- assumptions being made

Do not start coding before establishing the above.

### 5. Do Not Silently Guess
If any architecture, business rule, contract, or requirement is unclear:
- stop implementation
- explicitly list the unclear points
- mark any temporary assumptions clearly

Do not silently invent business logic.

### 6. Preserve Existing Contracts
Do not casually break:
- API contracts
- database compatibility
- auth flows
- event payloads
- shared component interfaces
- environment variable names
- external integration behavior

If a breaking change is required, flag it explicitly.

### 7. SaaS Safety Requirements
For every phase, check whether implementation affects:
- authentication
- authorization
- tenant isolation
- billing
- audit logs
- analytics
- background jobs
- notifications
- external integrations
- admin operations
- production rollout safety

### 8. Multi-Tenant Awareness
If the product is multi-tenant:
- all data access must be tenant-scoped
- caches must be tenant-safe
- background jobs must be tenant-safe
- storage paths must be tenant-safe
- permissions must be tenant-aware

### 9. Security by Default
For every feature, verify:
- input validation
- output safety
- permission checks
- secret handling
- PII exposure risk
- rate limiting needs
- abuse/failure scenarios

### 10. Prefer Existing Patterns
- Reuse existing architecture and patterns in the repo.
- Prefer shared components/utilities/modules over introducing new patterns.
- Add new dependencies only when necessary and justify them.

### 11. Testing Awareness
For each phase, include:
- what was tested
- what still needs testing
- happy paths
- edge cases
- regression risks

### 12. Migration and Deployment Awareness
When relevant, explicitly state:
- schema migrations
- data migrations
- seed data needs
- backward compatibility concerns
- rollout risks
- rollback considerations

### 13. Observability Awareness
When relevant, include:
- logs
- metrics
- tracing hooks
- audit entries
- error handling strategy
- retry/idempotency behavior

### 14. Document New Config
Whenever adding or changing:
- env vars
- feature flags
- queue configs
- cron configs
- integration configs

document them in the implementation summary.

---

## Required Output Format for Every Phase

At the end of each implementation phase, generate the following:

### A. Summary
- What was implemented
- Why it was implemented this way
- Key design decisions

### B. Changed Surface Area
- files changed
- modules affected
- APIs/contracts added or modified
- DB/schema changes if any

### C. Assumptions / Unclear Areas
- architecture doubts
- spec gaps
- temporary assumptions made

### D. Checklist
A concrete checklist another coding agent can use to validate completion.

Example:
- [ ] API route created
- [ ] service layer implemented
- [ ] validation added
- [ ] authorization enforced
- [ ] tenant scoping verified
- [ ] tests added/updated
- [ ] logs/errors handled
- [ ] env/config documented

### E. Handoff Context
A compact block that includes:
- current phase status
- what is already implemented
- important constraints
- known issues
- what the next agent should do next

This handoff must be concise and readable by any coding agent without needing to re-read the full implementation history.

---

## Required Stop Conditions

Stop and report instead of proceeding if:
- architecture contradicts itself
- core business logic is undefined
- data ownership is unclear
- tenant boundaries are unclear
- auth/permission logic is unclear
- API contract expectations are unclear
- schema changes may cause breakage but are unspecified
- external integration behavior is not defined

---

## Preferred Phase Size

Each phase should ideally target one of the following:
- one feature slice
- one service/module
- one API + service + persistence path
- one UI flow
- one background job flow
- one migration unit

Avoid combining many unrelated concerns into one phase.

---

## Implementation Philosophy

The objective is not only to generate code, but to generate:
- correct code
- reviewable code
- safe SaaS code
- handoff-friendly code
- context-window-efficient code
- production-aware code

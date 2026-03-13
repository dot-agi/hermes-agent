# Research Swarm: Fusing Agent-of-Agents with Multi-Agent Research Swarm

> Architecture, findings, analysis, and implementation plan for building a heterogeneous multi-agent research system on top of hermes-agent --- combining different backends, models, and specialized agents to autonomously perform R&D, run experiments, write code, iterate, and produce novel research.

---

## Table of Contents

- [1. Problem Statement](#1-problem-statement)
- [2. Source Ideas (from HERMES_AGENT_INNOVATIVE_USES.md)](#2-source-ideas)
- [3. Findings: hermes-agent Internal Capabilities](#3-findings-hermes-agent-internal-capabilities)
- [4. Findings: Multi-Agent Architecture Patterns](#4-findings-multi-agent-architecture-patterns)
- [5. Findings: Autonomous Research Loops (Karpathy's autoresearch)](#5-findings-autonomous-research-loops)
- [6. Analysis: What Hermes Can Do Today vs. What Needs Building](#6-analysis-what-hermes-can-do-today-vs-what-needs-building)
- [7. The Fused Architecture: Research Swarm](#7-the-fused-architecture-research-swarm)
- [8. Implementation Plan](#8-implementation-plan)
- [9. Observations & Open Questions](#9-observations--open-questions)

---

## 1. Problem Statement

We want to build a system on hermes-agent that:

1. **Integrates different backends** --- Modal (GPU), E2B, local, Docker, SSH --- so different agents can use different compute environments simultaneously
2. **Allows combination of models** --- different agents use different LLMs (e.g., Claude for reasoning, DeepSeek for code, Gemini for fast summarization) in the same workflow
3. **Performs R&D, experiments, writes codebase, writes papers, iterates** --- a full autonomous research loop
4. **Does novel and innovative research** --- not just task execution, but hypothesis generation, experimental design, and knowledge synthesis

This requires fusing two ideas from `HERMES_AGENT_INNOVATIVE_USES.md`:

- **#25 Agent-of-Agents**: A meta-agent orchestrating a fleet of specialized hermes instances with different toolsets and skills
- **#12 Multi-Agent Research Swarm**: Specialized subagents (Scholar, Industry, Code, Contrarian) researching from different angles, synthesized by a parent

---

## 2. Source Ideas

### Agent-of-Agents (Idea #25)

A meta-agent that manages a fleet of specialized hermes-agent instances, routing tasks to the right specialist and coordinating complex workflows. Each instance has different toolsets, skills, and roles. The orchestrator routes, coordinates, and tracks which specialist is best at which task type over time.

### Multi-Agent Research Swarm (Idea #12)

Submit a research question. The agent spawns specialized subagents that independently research from different angles (literature, industry, code, contrarian), then synthesizes findings using mixture-of-agents for balanced perspective. Skills capture research methodology per domain.

### The Fusion Goal

Combine the *orchestration layer* of Agent-of-Agents with the *research-specific specialization* of the Research Swarm, and add:
- Heterogeneous compute backends (GPU for experiments, local for coordination, Docker for safe code execution)
- Heterogeneous models (different LLMs for different cognitive tasks)
- An autonomous experiment loop (inspired by Karpathy's `autoresearch`)
- A knowledge accumulation system (skills + memory + shared state)

---

## 3. Findings: hermes-agent Internal Capabilities

### 3.1 Delegation System (`tools/delegate_tool.py`)

| Capability | Details |
|---|---|
| **Max concurrent subagents** | Configurable via `delegation.max_concurrent_children` (default: 3) |
| **Max depth** | Configurable via `delegation.max_depth` (default: 2) |
| **Per-subagent toolsets** | Yes --- each task in `tasks` array can specify its own `toolsets` |
| **Per-subagent model** | Yes --- each task can specify `model` + `provider` with graceful fallback to delegation defaults |
| **Per-subagent backend** | Yes --- each task can specify `terminal_backend` (local, docker, modal, ssh, singularity, daytona) |
| **Per-subagent iteration budget** | Yes --- each task can specify `max_iterations` for an independent budget, preventing starvation. Falls back to shared parent budget if unset |
| **Blocked tools for children** | `delegate_task`, `clarify`, `memory`, `send_message`, `execute_code` |
| **Communication** | Children return results to parent as JSON strings; no peer-to-peer (use blackboard skill for shared state) |

**Remaining limitations:** Children cannot write to memory (blocked to prevent concurrent write corruption --- use the blackboard skill for shared state instead). ~~Max depth 2~~ and ~~shared iteration budget~~ are resolved: depth is configurable via `delegation.max_depth`, and per-task `max_iterations` gives each child its own independent `IterationBudget` to prevent starvation.

### 3.2 Terminal Backend System (`tools/environments/`)

| Backend | Class | Key Capability |
|---|---|---|
| `local` | `LocalEnvironment` | Direct host execution |
| `docker` | `DockerEnvironment` | Isolated containers, security hardening, volume mounts |
| `ssh` | `SSHEnvironment` | Remote server execution, ControlMaster persistence |
| `modal` | `ModalEnvironment` | Serverless GPU, snapshot/restore filesystem |
| `singularity` | `SingularityEnvironment` | HPC containers |
| `daytona` | `DaytonaEnvironment` | Cloud dev environments |

**Factory function:** `_create_environment(env_type, image, cwd, timeout, ...)` --- called lazily per `task_id`.

**Per-task override:** `register_task_env_overrides(task_id, overrides)` allows different container images per task_id on the *same* backend. This is used by batch_runner and RL environments.

~~**Critical gap:**~~ **RESOLVED.** Per-task `terminal_backend` overrides are now supported. `terminal_tool.py` checks `overrides["terminal_backend"]` before the global `TERMINAL_ENV`, so subagents in the same process can use different backends.

### 3.3 Model Switching (`hermes_cli/auth.py`, `runtime_provider.py`)

- `AIAgent` accepts `base_url`, `api_key`, `provider`, `model` directly
- `delegation.model` + `delegation.provider` config fields control subagent models
- Auxiliary models configurable per-task: `auxiliary.vision`, `auxiliary.compression`, `auxiliary.web_extract`, etc.
- OpenRouter provider routing: `sort`, `only`, `ignore`, `order`, `require_parameters`

~~**Gap:**~~ **RESOLVED.** Each task in the `tasks` array can specify its own `model` and `provider`. `_resolve_task_creds()` resolves per-task credentials with graceful fallback to delegation defaults.

### 3.4 Batch Runner (`batch_runner.py`)

- Uses `multiprocessing.Pool` --- each worker is a **separate OS process**
- Per-prompt container image override via dataset `"image"` field
- Toolset sampling via named distributions (probabilistic per-toolset)
- Automatic checkpointing and resume
- Trajectory export in ShareGPT format

**Key insight:** Batch runner already solves the "different backends per task" problem by running separate OS processes. Each worker process can have its own `TERMINAL_ENV`.

### 3.5 Execute Code (`tools/code_execution_tool.py`)

- Unix Domain Socket RPC: child Python script calls hermes tools via `hermes_tools.web_search()` etc.
- Sandbox-allowed tools: `web_search`, `web_extract`, `read_file`, `write_file`, `search_files`, `patch`, `terminal`
- 5-minute timeout, 50 tool calls max, 50KB stdout
- `execute_code` iterations are refunded from IterationBudget (don't burn budget)

### 3.6 RL Environments (`environments/`)

- `HermesAgentBaseEnv` + `HermesAgentLoop` for Atropos integration
- 11 tool-call parsers for different model architectures
- Per-rollout `task_id` with sandbox isolation
- `ToolContext` gives reward functions access to the same sandbox as the model
- Two-phase operation: Phase 1 (OpenAI, structured tool calls) vs Phase 2 (VLLM, raw text + parsing)

### 3.7 Toolset System (`toolsets.py`)

- 20+ individual toolsets, composable into presets
- `create_custom_toolset()` for runtime mutation
- `resolve_multiple_toolsets()` for union composition
- Distributions for probabilistic toolset sampling (batch/RL)

---

## 4. Findings: Multi-Agent Architecture Patterns

Based on the Openlayer multi-agent architecture guide (March 2026) and broader research:

### Applicable Patterns

| Pattern | Fit for Research Swarm | Why |
|---|---|---|
| **Hierarchical** | Primary pattern | Research naturally decomposes into phases (literature review → hypothesis → experiment → analysis → writing), each managed by a mid-level supervisor |
| **Blackboard** | Knowledge sharing | A shared knowledge base (research log, findings database) where specialist agents read partial results and contribute refinements |
| **Supervisor** | Per-phase coordination | Within each phase, a supervisor coordinates the specialist workers |
| **Swarm** | Experiment exploration | For hyperparameter search and hypothesis testing, swarm-style parallel exploration (a la Karpathy's autoresearch) |

### Key Insight from Research

> "Single agents hit a ceiling at 10-15 tools. Multi-agent systems outperform single agents on parallelizable tasks but degrade performance by 39-70% on sequential reasoning." --- Openlayer, citing Google research

This means: use multi-agent for *parallel research and experimentation*, but use single-agent for *sequential reasoning and paper writing*.

---

## 5. Findings: Autonomous Research Loops

### Karpathy's `autoresearch` Pattern (March 2026)

A 630-line MIT-licensed script that implements:

1. **Read** the current code/config
2. **Hypothesize** an improvement
3. **Modify** the code
4. **Run** an experiment (fixed compute budget, e.g., 5 min GPU)
5. **Evaluate** --- if metric improves, keep; otherwise revert
6. **Loop** indefinitely

Results: 126 autonomous experiments overnight, improved validation loss from 0.9979 to 0.9697. Later, 700 autonomous changes over 2 days, ~20 additive improvements that transferred to larger models.

### Hyperspace Swarm Extension

35 agents ran 333 unsupervised experiments on a peer-to-peer network. Key emergent behaviors:
- **Hardware diversity as a feature**: GPU agents explored brute-force; CPU agents explored strategic lever changes (initialization, normalization)
- **Gossip protocol for knowledge sharing**: When one agent found that Kaiming init reduced loss 21%, it propagated across the network and 23 other agents built on it within hours
- **Compressed history**: The swarm rediscovered RMSNorm and tied embeddings in 17 hours --- techniques that took human researchers ~8 years to formalize

### Relevance to Our Design

The autoresearch loop maps directly onto hermes-agent's capabilities:
- **Read/Hypothesize** → Agent reasoning (any LLM)
- **Modify** → `write_file` / `patch` tools
- **Run** → Terminal backend (Modal for GPU, Docker for CPU)
- **Evaluate** → `execute_code` for metric extraction
- **Knowledge sharing** → Skills system + shared Database MCP (our blackboard)
- **Loop** → Cron or continuous agent loop

---

## 6. Analysis: What Hermes Can Do Today vs. What Needs Building

### Works Today (No Code Changes)

| Capability | How |
|---|---|
| Spawn 3 parallel research subagents | `delegate_task` with `tasks` array |
| Different toolsets per subagent | `toolsets` field in each task |
| Single-model delegation | `delegation.model` + `delegation.provider` in config |
| Sequential research phases | Parent orchestrates: Phase 1 → Phase 2 → Phase 3 |
| Persistent knowledge accumulation | Memory (MEMORY.md) + Skills (SKILL.md) |
| Experiment execution on GPU | Modal backend |
| Safe code execution | Docker backend |
| Scheduled experiment runs | Cron system |
| Cross-session recall | Session search (FTS5 + LLM summarization) |
| Trajectory generation for training | Batch runner |
| Multi-model synthesis | `mixture_of_agents` tool |
| Paper writing | Agent reasoning + file tools |

### Requires External Orchestration (No Core Changes, But Needs a Wrapper)

| Capability | Gap | Solution |
|---|---|---|
| ~~Different models per subagent~~ | ~~`delegate_task` uses same model for all tasks~~ | **RESOLVED:** per-task `model` + `provider` fields in task schema |
| ~~Different backends per subagent~~ | ~~`TERMINAL_ENV` is process-global~~ | **RESOLVED:** per-task `terminal_backend` field in task schema |
| Peer-to-peer knowledge sharing between agents | Children return to parent only, no peer communication | **Blackboard skill** wrapping shared SQLite --- all agents read/write via `terminal` |
| Continuous experiment loop (autoresearch-style) | No built-in loop-until-converge primitive | **Experiment loop skill** with helper scripts for metric tracking + git checkpointing |
| ~~More than 3 parallel agents~~ | ~~`MAX_CONCURRENT_CHILDREN = 3`~~ | **RESOLVED:** configurable via `delegation.max_concurrent_children` |

### Would Benefit from Core Enhancements (Optional, But Powerful)

| Capability | Enhancement |
|---|---|
| ~~Per-task model in `delegate_task`~~ | **IMPLEMENTED:** `model` + `provider` fields on per-task schema items |
| ~~Per-task backend in `delegate_task`~~ | **IMPLEMENTED:** `terminal_backend` field on per-task schema items |
| ~~Configurable delegation limits~~ | **IMPLEMENTED:** `delegation.max_depth` and `delegation.max_concurrent_children` in config |
| Shared blackboard | **Skill** (not core tool) --- `optional-skills/research/blackboard/` with SQLite helper scripts |
| Experiment loop | **Skill** (not core tool) --- `optional-skills/research/experiment-loop/` with loop protocol instructions |

---

## 7. The Fused Architecture: Research Swarm

### 7.1 Overview

```
┌─────────────────────────────────────────────────────┐
│                ORCHESTRATOR AGENT                    │
│  Model: Claude Opus (best reasoning)                │
│  Backend: Local                                     │
│  Role: Decompose research goal, coordinate phases,  │
│        synthesize final output, write paper          │
│  Tools: memory, skills, delegate_task, mixture_of_  │
│         agents, execute_code, web_search             │
└────────┬──────────────┬──────────────┬──────────────┘
         │              │              │
    Phase 1        Phase 2        Phase 3
    SURVEY         EXPERIMENT     SYNTHESIS
         │              │              │
┌────────▼────┐  ┌──────▼──────┐  ┌───▼──────────┐
│ LITERATURE  │  │ EXPERIMENT  │  │ WRITING      │
│ SUPERVISOR  │  │ SUPERVISOR  │  │ SUPERVISOR   │
│ Model:      │  │ Model:      │  │ Model:       │
│  Gemini     │  │  DeepSeek   │  │  Claude      │
│  Flash      │  │  Coder      │  │  Sonnet      │
│ Backend:    │  │ Backend:    │  │ Backend:     │
│  Local      │  │  Modal(GPU) │  │  Local       │
│             │  │  Docker     │  │              │
└──┬───┬───┬──┘  └──┬───┬───┬──┘  └──┬───┬──────┘
   │   │   │        │   │   │        │   │
   ▼   ▼   ▼        ▼   ▼   ▼        ▼   ▼
 Workers:          Workers:          Workers:
 - Scholar         - Coder           - Drafter
 - Industry        - Runner          - Reviewer
 - Contrarian      - Analyzer        - Formatter
```

### 7.2 The Blackboard: Shared Research State

All agents share state via a **SQLite database** accessed through the MCP filesystem or a custom Database MCP server. This is the blackboard.

```
research_blackboard.db
├── hypotheses        # id, text, status, source_agent, evidence_for, evidence_against
├── experiments       # id, hypothesis_id, code_path, config, status, result, metric
├── findings          # id, source (paper/experiment/web), summary, relevance_score
├── knowledge_graph   # entity, relation, entity, source_finding_id
├── paper_sections    # section_name, content, version, last_updated_by
└── experiment_log    # id, timestamp, agent, action, outcome, notes
```

**How it works:**
1. Literature agents write to `findings` and `knowledge_graph`
2. Experiment agents read `hypotheses`, run experiments, write to `experiments`
3. The orchestrator reads all tables, decides next phase, updates `hypotheses`
4. Writing agents read `findings` + `experiments`, draft `paper_sections`
5. Every agent can see what every other agent has discovered

### 7.3 Phase 1: Literature Survey (Parallel, Fast Models)

**Goal:** Understand the state of the art. What's been tried? What works? What's the gap?

**Model selection:** Gemini Flash or similar fast/cheap model (via OpenRouter) --- literature survey is high-volume, low-reasoning.

**Subagents (3 parallel):**

| Agent | Toolsets | Task |
|---|---|---|
| **Scholar** | `web`, `browser` | Search arxiv, Google Scholar, Semantic Scholar. Extract methods, results, limitations. Write to `findings` |
| **Industry** | `web`, `browser` | Search company blogs, GitHub repos, HuggingFace. Find implementations, benchmarks, practical results. Write to `findings` |
| **Contrarian** | `web`, `browser` | Specifically search for failures, limitations, negative results, replication failures. Write to `findings` with `evidence_against` links |

**Output:** Populated `findings` table + `knowledge_graph` + a literature summary in `paper_sections["related_work"]`.

### 7.4 Phase 2: Experimentation (Heterogeneous Backends, Code-Focused Models)

**Goal:** Form hypotheses, write code, run experiments, iterate.

This phase implements the **autoresearch loop** inside hermes-agent:

**Model selection:** DeepSeek Coder or Qwen Coder (via OpenRouter) for code generation. Code-focused models are cheaper and better at code than general models.

**Subagents (3 parallel, cascaded if needed):**

| Agent | Backend | Task |
|---|---|---|
| **Coder** | Docker | Reads `hypotheses` from blackboard. Writes experiment scripts. Commits to a git worktree |
| **Runner** | Modal (GPU) | Executes experiment scripts within fixed compute budgets. Writes results to `experiments` table |
| **Analyzer** | Local | Reads `experiments` results. Performs statistical analysis via `execute_code`. Updates `hypotheses` (confirmed/refuted). Suggests next experiments |

**The Experiment Loop:**

```
while budget_remaining and not converged:
    1. Analyzer reads blackboard → generates hypothesis
    2. Coder writes experiment code → commits to worktree
    3. Runner executes on Modal (GPU) → writes results to blackboard
    4. Analyzer evaluates results → updates hypothesis status
    5. If improved: keep changes, record skill
    6. If not: revert, try different hypothesis
    7. Knowledge (successful patterns) shared via blackboard
```

**Implementing across backends without core changes:**

Since `TERMINAL_ENV` is process-global, the experiment loop uses `execute_code` as the bridge:

```python
# The Orchestrator's execute_code script for Phase 2:
import hermes_tools
import json

# 1. Read current hypotheses from blackboard
result = hermes_tools.terminal("sqlite3 research_blackboard.db 'SELECT * FROM hypotheses WHERE status=\"active\"'")
hypotheses = parse_hypotheses(result)

# 2. For each hypothesis, write experiment code
for h in hypotheses:
    # Generate code (this call goes to the LLM via RPC)
    code = hermes_tools.web_search(f"implementation of {h.method}")  # research
    hermes_tools.write_file(f"experiments/{h.id}/train.py", experiment_code)

    # 3. Run on Modal (via terminal which uses current backend)
    result = hermes_tools.terminal(f"python experiments/{h.id}/train.py")

    # 4. Record result
    hermes_tools.terminal(f"sqlite3 research_blackboard.db \"INSERT INTO experiments ...\"")
```

**Alternative: Multi-process orchestration** for true backend heterogeneity:

```bash
# orchestrate.sh - launch separate hermes instances per backend
TERMINAL_ENV=docker hermes --quiet -Q "Run the coder agent tasks from blackboard" &
TERMINAL_ENV=modal hermes --quiet -Q "Run the experiment scripts from blackboard on GPU" &
TERMINAL_ENV=local hermes --quiet -Q "Analyze experiment results in blackboard" &
wait
```

### 7.5 Phase 3: Synthesis & Paper Writing (Strong Reasoning Model)

**Goal:** Synthesize all findings into a coherent paper/report.

**Model selection:** Claude Opus or GPT-4o (best reasoning, best writing).

**Subagents (2 parallel):**

| Agent | Task |
|---|---|
| **Drafter** | Reads `findings` + `experiments` + `knowledge_graph` from blackboard. Writes each `paper_sections` entry: abstract, introduction, methods, results, discussion, conclusion |
| **Reviewer** | Reads draft sections. Checks claims against evidence in blackboard. Flags unsupported claims. Suggests improvements. Acts as internal peer reviewer |

**Iteration:** Drafter and Reviewer alternate until the Reviewer's objection count drops below threshold.

**Final output:** Complete paper/report as markdown + LaTeX, with all experiments reproducible from the git worktree.

### 7.6 Model Routing Strategy

| Cognitive Task | Best Model Type | Why | Config |
|---|---|---|---|
| Orchestration & decomposition | Claude Opus | Best at complex reasoning, planning | Main agent model |
| Literature survey (high volume) | Gemini Flash | Fast, cheap, good at summarization | `delegation.model` for Phase 1 |
| Code generation | DeepSeek Coder / Qwen Coder | Best at code, cheaper | `delegation.model` for Phase 2 |
| Statistical analysis | Claude Sonnet | Good reasoning, cheaper than Opus | `auxiliary.compression` model reused |
| Paper writing | Claude Opus | Best writing quality | Main agent (Phase 3, no delegation) |
| Multi-perspective synthesis | Mixture of Agents | Queries 3+ models, synthesizes | `mixture_of_agents` tool |
| Quick fact-checking | Gemini Flash | Fast, cheap | `auxiliary.web_extract` |

**Implementation without core changes:**

The orchestrator swaps `delegation.model` and `delegation.provider` in `~/.hermes/config.yaml` between phases:

```python
# In orchestrator's execute_code script:
import yaml, hermes_tools

def set_delegation_model(model, provider="openrouter"):
    """Swap the delegation model between phases."""
    config_path = os.path.expanduser("~/.hermes/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config.setdefault("delegation", {})["model"] = model
    config["delegation"]["provider"] = provider
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

# Phase 1: Fast model for literature
set_delegation_model("google/gemini-2.0-flash-001")
hermes_tools.terminal("hermes -Q 'Run Phase 1: literature survey'")

# Phase 2: Code model for experiments
set_delegation_model("deepseek/deepseek-coder-v3")
hermes_tools.terminal("hermes -Q 'Run Phase 2: experiments'")

# Phase 3: Strong model for writing
set_delegation_model("anthropic/claude-opus-4-20250514")
hermes_tools.terminal("hermes -Q 'Run Phase 3: write the paper'")
```

### 7.7 Backend Integration Strategy

| Phase | Backend | Why |
|---|---|---|
| Orchestrator | Local | Lightweight, fast, needs full config access |
| Literature Survey | Local | Web search + browser, no heavy compute |
| Code Writing | Docker | Isolated, safe code execution, git worktree |
| Experiment Running | Modal | GPU access, serverless scaling, snapshot/restore |
| Statistical Analysis | Local | Python + libraries, no GPU needed |
| Paper Writing | Local | Reasoning-heavy, no compute needed |

**For true multi-backend in a single workflow** (no core changes needed):

```yaml
# ~/.hermes/config.yaml - use Docker as default
terminal:
  backend: docker
  container_image: "python:3.11-slim"

# For GPU experiments, spawn a separate hermes process:
# TERMINAL_ENV=modal hermes --quiet -Q "run experiments/train.py"
```

Or use the **batch_runner pattern**: the batch runner already spawns separate OS processes per worker, and per-prompt image overrides are supported. A research experiment dataset can specify `"image": "nvidia/cuda:12.4-devel"` for GPU tasks and `"image": "python:3.11-slim"` for analysis tasks.

### 7.8 Scaling Beyond 3 Agents: The Cascade

`delegate_task` caps at 3 concurrent children. For more parallelism, use hierarchical delegation:

```
Orchestrator
├── Literature Supervisor (delegate_task)
│   ├── Scholar Agent
│   ├── Industry Agent
│   └── Contrarian Agent
├── Experiment Supervisor (delegate_task)
│   ├── Coder Agent
│   ├── Runner Agent
│   └── Analyzer Agent
└── Writing Supervisor (delegate_task)
    ├── Drafter Agent
    ├── Reviewer Agent
    └── Formatter Agent
```

This gives 9 parallel workers across 3 supervisors. The limitation is `MAX_DEPTH = 2`, so supervisors are the final level --- they cannot further delegate. This is sufficient for most research workflows.

---

## 8. Implementation Plan

### Phase 0: Foundation (Week 1)

**Goal:** Set up the blackboard and validate basic multi-agent delegation.

1. **Create the blackboard database schema**
   - Create `research_blackboard.db` with tables: `hypotheses`, `experiments`, `findings`, `knowledge_graph`, `paper_sections`, `experiment_log`
   - Write a helper Python module (`blackboard.py`) with read/write functions
   - Test via `execute_code`

2. **Validate delegation with different toolsets**
   - Create a test where 3 subagents with different toolsets (web, browser, terminal) run in parallel
   - Verify results are returned to parent
   - Measure iteration budget consumption

3. **Validate model switching between phases**
   - Write an `execute_code` script that swaps `delegation.model` in config
   - Run Phase 1 with Gemini Flash, Phase 2 with DeepSeek, verify each uses correct model

### Phase 1: Literature Swarm (Week 2)

**Goal:** Automated literature survey that populates the blackboard.

1. **Build the Scholar agent skill**
   - Input: research topic/question
   - Tools: `web_search`, `browser_navigate`, `browser_snapshot`
   - Output: structured findings in blackboard `findings` table
   - Captures: paper title, authors, year, methods, results, limitations, relevance score

2. **Build the Industry agent skill**
   - Same structure but targets: GitHub repos, HuggingFace models, blog posts, company announcements
   - Captures: implementation URLs, benchmark results, production deployment notes

3. **Build the Contrarian agent skill**
   - Specifically searches for: replication failures, negative results, critiques, limitations
   - Flags contradictions between findings from Scholar/Industry agents

4. **Build the Literature Supervisor**
   - Receives research question
   - Delegates to 3 agents in parallel
   - Waits for completion
   - Synthesizes findings into a literature review (writes to `paper_sections["related_work"]`)
   - Identifies research gaps → writes initial `hypotheses`

### Phase 2: Experiment Loop (Week 3-4)

**Goal:** Autonomous hypothesis → code → run → evaluate → iterate loop.

1. **Build the Coder agent skill**
   - Input: hypothesis from blackboard
   - Reads related findings for context
   - Writes experiment script (Python + dependencies)
   - Commits to git worktree for isolation
   - Output: path to experiment script + config

2. **Build the Runner agent skill**
   - Input: experiment script path + config
   - Executes within compute budget (configurable: 5 min, 30 min, etc.)
   - Captures: metrics (loss, accuracy, etc.), runtime, resource usage
   - Writes results to `experiments` table
   - On Modal: uses GPU; on Docker: uses CPU

3. **Build the Analyzer agent skill**
   - Input: experiment results from blackboard
   - Performs statistical analysis (significance testing, effect sizes)
   - Compares against baseline and prior experiments
   - Updates `hypotheses` status: confirmed / refuted / inconclusive
   - Generates next hypotheses based on patterns
   - Writes analysis to `experiment_log`

4. **Build the Experiment Supervisor**
   - Orchestrates the Coder → Runner → Analyzer loop
   - Implements convergence criteria: stop when metric plateaus or budget exhausted
   - Tracks which hypotheses have been tested
   - Reports progress to orchestrator

5. **Implement multi-backend execution**
   - Option A: `execute_code` script that shells out to `TERMINAL_ENV=modal hermes ...`
   - Option B: Separate runner process with `TERMINAL_ENV=modal` managed by cron/supervisor
   - Option C: Use batch_runner with per-prompt image overrides for parallel experiments

### Phase 3: Paper Writing (Week 5)

**Goal:** Synthesize everything into a coherent paper.

1. **Build the Drafter agent skill**
   - Reads all blackboard tables
   - Writes each section: abstract, introduction, related work, methods, experiments, results, discussion, conclusion
   - Includes proper citations from `findings`
   - Includes tables/figures descriptions from `experiments`

2. **Build the Reviewer agent skill**
   - Reads draft sections
   - Cross-references claims against blackboard evidence
   - Flags: unsupported claims, missing citations, logical gaps, unclear writing
   - Provides specific improvement suggestions

3. **Build the Writing Supervisor**
   - Orchestrates Drafter → Reviewer iteration
   - Stops when Reviewer objection count < threshold (e.g., < 3 minor issues)
   - Produces final paper in markdown + optional LaTeX conversion

### Phase 4: Integration & Continuous Research (Week 6)

**Goal:** Wire everything together into a single command.

1. **Build the Orchestrator skill**
   - Input: research question + compute budget + model preferences
   - Runs Phase 1 → Phase 2 → Phase 3 sequentially
   - Swaps models between phases
   - Delivers final paper via messaging gateway (Telegram/Slack/Email)
   - Saves the research workflow as a reusable skill

2. **Build continuous research mode**
   - Use cron to schedule daily research iterations
   - Each day: check for new papers (Scholar agent), run follow-up experiments, update paper draft
   - Memory tracks research progress across sessions
   - Skills accumulate methodology improvements

3. **Build the meta-research loop**
   - After each research cycle, the orchestrator evaluates: "Did this workflow produce good research?"
   - Updates skills with methodological improvements
   - Adjusts model routing, compute budgets, and agent configurations
   - Over time, the system gets better at doing research

### Core Enhancements (IMPLEMENTED)

The following fixes have been applied to the codebase:

1. **Per-task model in delegate_task** --- `model` and `provider` fields added to per-task schema items in `delegate_tool.py`. The `_resolve_task_creds()` helper resolves per-task credentials with graceful fallback to delegation defaults.
2. **Per-task backend** --- `terminal_backend` field added to per-task schema (enum: local, docker, modal, ssh, singularity, daytona). `_setup_task_backend()` registers per-task overrides via `register_task_env_overrides()`, and `terminal_tool.py` now honors the `terminal_backend` override before reading the global `TERMINAL_ENV`.
3. **Configurable MAX_DEPTH and MAX_CONCURRENT_CHILDREN** --- no longer hardcoded. Configurable via `delegation.max_depth` and `delegation.max_concurrent_children` in config (defaults: 2 and 3 respectively).

### Blackboard: Skill, Not Core Tool

After analysis, a core `blackboard` tool is unnecessary. Three existing primitives already support shared state between agents:

- **SessionDB (WAL SQLite)** --- all `delegate_task` children are threads sharing the parent's `session_db` reference (`delegate_tool.py:257`). WAL mode supports concurrent readers + one writer. A parent can create a "blackboard session" before dispatching, and children read/write via the shared filesystem.
- **Skill with `scripts/blackboard.py`** --- a skill ships helper scripts that manage a SQLite DB at a known path. The SKILL.md instructs the LLM to init/read/write/query via `terminal("python3 SKILL_DIR/scripts/blackboard.py ...")`. This gives schema enforcement, atomic writes, and structured queries without core changes.
- **`execute_code` RPC + shared filesystem** --- orchestrator scripts use `hermes_tools.terminal("sqlite3 blackboard.db '...'")` to read/write structured data. Delegate children do the same via their `terminal` tool.

**Recommended approach:** Implement as an `optional-skill` with:
```
optional-skills/research/blackboard/
├── SKILL.md              # Protocol: init, read, write, query, subscribe
├── scripts/
│   └── blackboard.py     # SQLite wrapper with schema enforcement
└── templates/
    └── schema.sql        # Default blackboard schema (hypotheses, experiments, findings, etc.)
```

This proves the pattern without core changes. If heavily used, it can be promoted to a core tool later.

### Experiment Loop: Skill, Not Core Tool

The proposed `run_experiment_loop(script, metric, budget, max_iterations)` core tool is also unnecessary. The experiment loop is a **protocol** (instruction sequence), not a **mechanism** (Python function). The LLM's own agentic loop IS the experiment loop --- each iteration is another turn of the agent loop.

**Why existing primitives suffice:**
- `execute_code` runs experiment scripts with RPC tool access
- `terminal` handles git operations (commit on improvement, revert on failure)
- `delegate_task` parallelizes independent experiments (with per-task model/backend overrides)
- The `subagent-driven-development` skill already demonstrates this pattern: implement → review → fix → repeat, as pure instructions with no core code
- Helper scripts in `scripts/` can enforce hard constraints (budget cutoffs, mandatory reverts) that the LLM instructions alone can't guarantee

**Where a core tool WOULD be justified:** Sub-second iteration speed (hundreds of experiments/minute) where LLM overhead per turn is unacceptable. Research experiments typically take minutes to hours, making LLM overhead negligible.

**Recommended approach:** Implement as an `optional-skill`:
```
optional-skills/research/experiment-loop/
├── SKILL.md              # Loop protocol: hypothesis → code → run → evaluate → iterate
├── scripts/
│   ├── experiment.py     # Run script, capture metric, compare baseline
│   └── git_checkpoint.py # Commit on improvement, revert on failure
└── templates/
    └── experiment_config.yaml  # Template for experiment definitions
```

Both skills depend on the blackboard skill for shared state, forming a composable research toolkit.

---

## 9. Observations & Open Questions

### Observations

1. **hermes-agent's primitives are surprisingly close to what's needed.** The delegation system (now with per-task model/backend overrides), toolset composition, batch runner, and execute_code provide ~95% of the infrastructure. The remaining gap --- peer-to-peer agent communication --- is solved by the blackboard pattern via shared SQLite on the filesystem, implementable as a skill.

2. **The blackboard pattern is the key architectural decision.** By using a shared SQLite database as the blackboard, we get around the limitation that subagents can't communicate with each other. They communicate *through the blackboard*.

3. **Model switching between phases is cheap and effective.** Literature survey doesn't need Opus-level reasoning. Code generation doesn't need Gemini's speed. Matching model capability to cognitive task type saves money and improves results.

4. **The autoresearch pattern (hypothesis → code → run → evaluate → iterate) maps cleanly onto hermes's execute_code + terminal + skills.** The 5-minute compute budget per experiment is a natural unit of work for Modal's serverless pricing.

5. **The skills system is the long-term memory of the research program.** Over multiple research cycles, the swarm accumulates: "for NLP experiments, always include a BPE tokenizer comparison" or "DeepSeek Coder is better at PyTorch code than TensorFlow." This is genuine meta-learning.

6. **Cascade delegation (3 supervisors × 3 workers) gives 9 effective parallel agents**, which is sufficient for most research workflows. If more are needed, the batch_runner (multiprocessing.Pool) scales to arbitrary worker counts.

7. **The `execute_code` refund mechanism is brilliant for research loops.** Since execute_code iterations don't consume the shared iteration budget, a single execute_code call can orchestrate hundreds of experiment iterations without burning the LLM budget.

8. **Skills are the right abstraction for blackboard and experiment loops.** Both are protocols (instruction sequences), not mechanisms (Python functions). The `subagent-driven-development` skill already proves that complex iterative loops work as pure LLM instructions. A core tool would be premature --- skills can prove the pattern first, then promote to core if adoption warrants it. This also makes them distributable via Skills Hub without requiring hermes-agent core releases.

### Open Questions

1. **Convergence criteria:** How does the experiment loop decide when to stop? Options: fixed iteration count, metric plateau detection, compute budget exhaustion, or human checkpoint.

2. **Hypothesis quality:** What prevents the system from testing trivial hypotheses? Options: Contrarian agent as filter, require literature support for each hypothesis, human approval for experiment plans.

3. **Reproducibility:** How do we ensure experiments are reproducible? Options: git worktree per experiment, Docker image snapshots, full environment logging.

4. **Cost management:** Running Opus for orchestration + DeepSeek for code + Modal for GPU can add up. Options: set hard dollar budgets per phase, use OpenRouter cost tracking, preference cheap models for routine tasks.

5. **Evaluation of research quality:** How do we know the output is good? Options: internal Reviewer agent, human review checkpoints, comparison against known baselines, external peer review.

6. **Knowledge transfer between research programs:** Can skills from one research topic transfer to another? This is an empirical question that the system itself could investigate.

7. **Safety of autonomous experimentation:** What prevents the experiment loop from generating harmful code or consuming excessive resources? Options: Docker sandboxing, compute budget caps, code review agent as gate, blocked dangerous commands.

---

*This document serves as both the analysis of the current hermes-agent architecture and the blueprint for building the Research Swarm on top of it. The phased implementation plan is designed so that each phase produces a usable intermediate result, and the system becomes increasingly autonomous with each phase.*

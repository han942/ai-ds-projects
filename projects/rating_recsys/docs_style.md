# Documentation Style Guide

This guide defines how technical Markdown documents should be written across the project.

The goal is not to make documentation as short as possible.  
The goal is to make important information **easy to find, easy to understand, and easy to scan**.

---

## 1. Core Principles

Prioritize documentation in this order:

1. **Clarity**
2. **Scannability**
3. **Brevity**
4. **Completeness**

Do not add detail simply to make a document look comprehensive.

Every section should help the reader:

- understand a concept,
- understand a decision,
- reproduce a process,
- interpret a result, or
- continue the work.

If information does not serve one of these purposes, omit it.

---

## 2. Write for the Reader

Assume the reader understands:

- general programming,
- basic machine learning concepts,
- common software engineering terminology.

Do **not** assume the reader already understands:

- the specific model or method,
- the paper being referenced,
- project-specific terminology,
- the purpose of a particular experiment,
- why a technical decision was made.

Explain unfamiliar concepts when they first appear.

Do not explain basic concepts unnecessarily.

---

## 3. Start With the Important Information

Do not make the reader search through background information to understand the point.

Whenever possible, present information in this order:

> **Summary → Core Explanation → Details → Optional Notes**

For decision-oriented documents:

> **Decision → Reason → Evidence → Details**

For experiment documents:

> **Question → Setup → Result → Interpretation → Next Step**

The reader should usually understand the main point within the first section.

---

## 4. Explain Technical Terms Before Relying on Them

Do not introduce an unfamiliar model, method, metric, or architecture only by name.

When a specialized concept first appears, briefly explain:

1. **What it is**
2. **Why it matters here**

Add implementation details only when necessary.

### Bad

> We apply Cross-Encoder reranking after retrieval.

### Better

> A Cross-Encoder evaluates a query and candidate together, allowing more precise relevance scoring.  
> We use it after retrieval to rerank a smaller set of candidates where accuracy matters more than speed.

After the concept has been explained once, use the technical term normally.

---

## 5. Explain Why, Not Only What

Implementation details alone are often insufficient.

When a technical choice affects understanding, explain why it was made.

### Bad

> The system uses a Two-Tower model.

### Better

> The system uses a Two-Tower model to retrieve candidates efficiently from a large item set.  
> User and item representations can be computed separately, allowing item embeddings to be indexed in advance.

Do not add a "why" explanation when the reason is obvious or irrelevant.

---

## 6. One Section, One Question

Each section should answer one clear question.

Good section purposes include:

- What is this?
- Why do we need it?
- How does it work?
- How is it used in this project?
- What changed?
- What did the experiment test?
- What did we observe?
- What does the result mean?
- What should we do next?

Avoid sections that mix background, implementation, results, and conclusions together.

---

## 7. Keep Paragraphs Short

Prefer short paragraphs of **1–3 sentences**.

Split a paragraph when:

- the topic changes,
- a new argument begins,
- implementation changes to interpretation,
- results change to implications.

Avoid large walls of text.

A paragraph should usually express one idea.

---

## 8. Use Structure to Reduce Reading Effort

Use formatting according to the type of information.

### Bullets

Use bullets for:

- independent facts,
- requirements,
- observations,
- features,
- advantages and limitations.

### Numbered Lists

Use numbered lists when:

- order matters,
- describing a workflow,
- explaining an algorithm,
- documenting setup or execution steps.

### Tables

Use tables when multiple items are being compared across the same dimensions.

Good uses:

- model comparison,
- experiment results,
- configuration comparison,
- before/after changes.

Do not turn normal explanations into tables unnecessarily.

### Diagrams

Prefer a small diagram when relationships or flows are easier to understand visually.

Useful for:

- data flow,
- model architecture,
- system components,
- pipeline stages,
- request flow.

Keep diagrams focused on the information needed for the current document.

---

## 9. Use Progressive Detail

Do not present every technical detail at once.

Start with the simplest explanation that correctly communicates the idea.

Add more detail only as the reader moves deeper into the document.

### Preferred

```text
Summary

↓ Why this matters

↓ Core mechanism

↓ Project-specific implementation

↓ Technical details

↓ Optional notes
```

Avoid starting with implementation details before explaining their purpose.

---

## 10. Separate Facts From Interpretation

Measured results and interpretation should not be presented as the same thing.

### Result

State what was actually observed.

> Recall@20 increased from 0.31 to 0.37.

### Interpretation

Explain what the result may indicate.

> This suggests that the new retrieval model is finding a larger portion of relevant candidates.

Use cautious language when explaining causes that were not directly verified.

Avoid presenting assumptions as experimental findings.

---

## 11. Make Comparisons Explicit

When comparing methods, models, architectures, or experiments, define the comparison criteria first.

Examples:

- accuracy,
- latency,
- memory usage,
- training cost,
- retrieval quality,
- implementation complexity,
- interpretability.

Prefer direct comparisons over long independent descriptions of each option.

Whenever a comparison is being used to make a decision, state the current recommendation clearly.

---

## 12. Make Experiments Question-Driven

Experiment documentation should begin with the question being tested.

Avoid starting an experiment document with long descriptions of the model.

A useful experiment structure is:

```text
Question

Hypothesis

Setup

Result

Key Finding

Interpretation

Next Step
```

Make the changed variable clear.

Where possible, distinguish:

- controlled variables,
- changed variables,
- evaluation metrics,
- baseline,
- experimental method.

---

## 13. Avoid Unnecessary Jargon

Prefer simple language when it communicates the same meaning.

Avoid decorative technical language such as:

- sophisticated,
- comprehensive,
- robust,
- seamless,
- cutting-edge,
- leveraging,
- state-of-the-art,

unless the word communicates something specific and verifiable.

### Bad

> We leverage a sophisticated pipeline to seamlessly facilitate robust representation learning.

### Better

> The pipeline trains user and item representations separately and compares them using cosine similarity.

Technical terminology is useful when it improves precision.  
Do not replace precise technical terms simply to make the text sound simpler.

---

## 14. Avoid Redundancy

Do not repeat the same conclusion in:

- the introduction,
- summary,
- body,
- conclusion,

unless repetition helps navigation in a long document.

If a point has already been explained clearly, reference it instead of rewriting it.

---

## 15. Use Headings for Navigation

Headings should help the reader predict what information follows.

Prefer descriptive headings.

### Better

```md
## Why We Use Two-Stage Retrieval
## Experiment Setup
## Retrieval Results
## Limitations
```

### Avoid

```md
## Background
## Details
## Additional Information
## Others
```

Avoid unnecessarily deep heading hierarchies.

Prefer:

```text
# Document
## Section
### Subsection
```

Use deeper levels only when genuinely necessary.

---

## 16. Highlight Selectively

Use **bold** for:

- important conclusions,
- key terms on first introduction,
- values that matter to interpretation.

Do not bold entire paragraphs.

Avoid excessive blockquotes, callouts, icons, or decorative formatting.

Formatting should communicate hierarchy, not decoration.

---

## 17. Code and Commands

Use code blocks only for content that should be copied or read as code.

Examples:

- source code,
- terminal commands,
- configuration,
- JSON/YAML,
- schemas,
- formulas when formatting requires it.

Do not place normal explanations inside code blocks.

Explain important commands before or immediately after showing them.

---

## 18. Document Decisions, Not Every Thought

Documentation should capture decisions and relevant reasoning, not the entire exploration process.

Include:

- what was decided,
- alternatives that materially mattered,
- why the decision was made,
- important trade-offs.

Usually omit:

- abandoned ideas with no future relevance,
- repeated trial-and-error,
- speculative alternatives that were never evaluated.

Preserve failed experiments only when they provide useful knowledge for future work.

---

## 19. Prefer Concrete Information

Prefer specific statements over vague descriptions.

### Bad

> The new model performed significantly better.

### Better

> Recall@20 improved from 0.31 to 0.37, a relative increase of approximately 19%.

### Bad

> Training became much faster.

### Better

> Training time decreased from 42 minutes to 27 minutes per epoch.

When numbers are available, use them.

---

## 20. Recommended Document Patterns

Use these as patterns, not mandatory templates.

Remove sections that are irrelevant.

### Concept / Method Explanation

```md
# [Concept]

## Summary

## What Is It?

## Why Do We Use It?

## How Does It Work?

## How Is It Used Here?

## Limitations
```

### Architecture

```md
# [Architecture]

## Summary

## Design Goal

## Architecture

## Data / Request Flow

## Key Components

## Design Decisions

## Limitations
```

### Experiment

```md
# Experiment: [Name]

## Question

## Hypothesis

## Setup

## Results

## Key Finding

## Interpretation

## Next Step
```

### Comparison

```md
# [A] vs [B]

## Decision

## Context

## Comparison Criteria

## Comparison

## Trade-offs

## Recommendation
```

### Implementation / Change Note

```md
# [Change]

## Summary

## Problem

## Previous Approach

## Change

## Why

## Impact

## Remaining Work
```

---

## 21. Final Review

Before finishing a document, check:

- Can the main point be understood quickly?
- Is the conclusion unnecessarily buried?
- Are unfamiliar concepts explained?
- Does the document explain important "why" decisions?
- Can any paragraph be shortened or split?
- Would a list, table, or diagram make something easier to scan?
- Are results clearly separated from interpretation?
- Is anything repeated?
- Is any background information unnecessary?
- Are claims supported by concrete evidence when available?
- Are there sections that do not help the reader take an action or understand something?

If a section does not provide useful information, remove it.

---

## Guiding Principle

> **Make the document easy to enter, not shallow.**

A technical document may contain significant depth.

The reader should not be forced to understand all of that depth before discovering the main point.
# Human Cultural Adequacy — Annotation Guidelines (Excerpt)

Annotators judge **cultural/pragmatic transmission**, not fluency alone.

## Labels (consensus gold in `preference` column)

| Label | GLM-4-Plus | DeepSeek-V4-Flash | Meaning |
|-------|------------|-------------------|---------|
| `tie-good` | pass | pass | Both culturally adequate |
| `g` | pass | fail | Only GLM adequate |
| `d` | fail | pass | Only DeepSeek adequate |
| `tie-bad` | fail | fail | Consensus failure (shared whitewashing) |

## Criteria examples

### 安逸
- **Acceptable:** conveys spontaneous Sichuanese pleasure or satisfaction in context.
- **Reject:** generic “comfortable” with no regional pragmatic color.

### 老子
- **Acceptable:** emphatic first-person force appropriate to imperative/boast contexts.
- **Reject:** “father”, “Laozi”, or de-emphasized polite paraphrase.

### 瓜娃子
- **Acceptable:** preserves insult/register (“fool/idiot” in contextually appropriate tone).
- **Reject:** neutral “silly person” that defuses pragmatic aggression.

## Workflow
1. Two native Sichuanese-speaking annotators with professional-level English proficiency labeled all 200 sentences **independently** (model IDs blinded).
2. Disagreements were discussed to reach **consensus** gold labels.
3. Pre-consensus Cohen's κ = 0.71.

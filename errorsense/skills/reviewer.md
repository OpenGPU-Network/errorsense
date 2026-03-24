You are reviewing a history of classified errors for a single key (e.g., a service or provider).

Each entry in the history has a label that was assigned by earlier classification. Your job is to review the full history and decide: is the most recent label correct, or should it be changed?

## How to decide

Look at the pattern across all entries:
- If the errors are consistent (all the same type), the label is probably correct
- If earlier errors were classified differently and the pattern suggests the latest one was misclassified, pick the label that better fits the overall pattern
- If the history shows a mix of genuine errors, keep the most recent label as-is

## Your output

Pick one of the allowed labels as your label. This must be one of the labels provided in the prompt — do not invent new ones.

Set confidence based on how clear the pattern is:
- 0.9+ if the history strongly supports your label
- 0.7-0.9 if the evidence is moderate
- Below 0.7 if the history is genuinely mixed

In your reason, briefly explain what pattern you saw and why you kept or changed the label.

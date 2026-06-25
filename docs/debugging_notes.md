# Debugging Notes

## Critical bug: MPS `non_blocking=True` silently corrupted training (2026-06-24)

This was the most important bug we hit, and catching it is why we trust the
final numbers. Worth being able to walk through end-to-end.

### Symptom
While speeding up the training loop, we switched the metric accumulators to stay
on-device and added `non_blocking=True` to the host→device copies. The very next
smoke test reported **impossible** numbers: ~96–97% clean-val accuracy after a
*single* epoch on only 1,500 from-scratch images. A real from-scratch model is
around ~15% there — so we treated this as a red flag, not a win.

### Investigation
1. Re-evaluated the saved weights on a **class-balanced** set: the model
   predicted **class 0 for every image** — true accuracy ~4.8% (chance). It had
   collapsed, even though training *printed* 88–97%.
2. Found the smoke val subset happened to be all-class-0 (a class-ordered
   slice) — a cosmetic red herring that inflated that one number, but it did not
   explain the *balanced* 0.93 OOD reading.
3. **Smoking gun:** loading the same weights and calling `train.py`'s own
   `evaluate()` returned **different inflated values on different calls** (0.965,
   then 0.675) for weights that actually predict all-zeros. Non-deterministic
   results on fixed weights ⇒ a **race condition**.

### Root cause
`tensor.to(device, non_blocking=True)` on the **MPS** backend can return *before*
the copy finishes and is not safely ordered with the next kernel (unlike CUDA
with pinned memory). So the **label tensor `y` was read as garbage**, which:

- inflated every accuracy reading (predictions compared against garbage labels), and — far worse —
- fed garbage into `criterion(logits, y)`, so the model **trained on corrupted targets and collapsed** to a constant prediction.

### Fix
Drop `non_blocking=True`; keep the genuinely useful half of the change
(on-device metric accumulation, `.item()`-synced once per epoch rather than
every step — which is safe and avoids serializing MPS).

```python
# WRONG on MPS — races, corrupts the labels fed into the loss:
x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
# Correct:
x, y = x.to(device), y.to(device)
```

After the fix the smoke test returned honest low values (no collapse) and
`evaluate()` became deterministic.

### Lessons / safeguards now in place
- **Distrust too-good numbers.** A from-scratch model at 96% in one epoch is a
  bug signal, not a result.
- **Cross-verify through an independent path.** Every headline number is
  re-checked by loading `weights.joblib` through the grader's own `predict.py`,
  which is separate from the training loop's `evaluate()`.
- **Check the prediction histogram.** Predictions spread across all 20 classes
  rule out the class-collapse failure mode (worst-class accuracy ~0.75–0.79 in
  the final model).
- A code comment now warns against `non_blocking=True` at that call site.

---

## Related gotcha (not a model bug): the "stall" was stdout buffering

Right after the fix, full runs appeared to **hang** — no epoch lines in the log
for minutes, and the process sat near 0% CPU. We nearly chased it as a
throughput problem. It wasn't: a benchmark showed a full epoch is ~58 s
(~300 img/s, GPU-bound). The cause was that **Python block-buffers stdout when
it's redirected to a file**, so the per-epoch prints were stuck in a buffer and
never flushed to `train.log`. Running with `python -u` (unbuffered) made the log
update live, and epochs appeared on time.

Takeaway: when monitoring a redirected log, run unbuffered (`python -u` /
`PYTHONUNBUFFERED=1`), and confirm "slow" against an actual throughput
measurement before assuming a performance bug.

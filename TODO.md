# Code Work Queue

The canonical cross-repository queue is `../docs/TODO.md`. Detailed experiment
evidence and the adopted contract live in `../docs/THREE_BAND_RERUN_MAP.md`.

- Review/publish the adopted Adam `0.0001` protocol v3 and timing report v3 changes.
- Run the separately approved disposable H100 timing check within a 15-minute total
  allocation. Use at most 560 seconds of compute, measure M5..M20 and a real main-M5
  shuffled-read slice, audit and export only small evidence, excluding `bulk/`.
  Reconcile the resulting estimate and full-run budget with the owner before launch.
- Production remains the fixed full-size fresh experiment after separate approval.
  Cross-machine continuation and further sizing/model searches are outside this work.

# Research Journal

A running, narrative log of the QAT research project -- the story of the
work, not the technical record of it. For the detailed technical logs (exact
numbers, bugs, fixes), see `docs/reports/`. This is the "how it actually
felt and what we were thinking" record, kept alongside it.

---

## 2026-08-29

We hit the first moment in this project where the honest answer to "how's
it going" wasn't a number, it was "we don't know." Qwen2.5-0.5B's very first
QAT run -- the *control*, the simplest, cheapest configuration in the whole
matrix, the one with no quantization overhead at all -- has been running
for over five hours and still isn't done. OPT-350M and Pythia-410M did this
exact same thing in 20 to 45 minutes. Something about this model, on this
laptop, right now, is taking ten times longer than it should, and after
three separate occurrences of this pattern across two different models
(Pythia's eval time on Day 2, Qwen's training time on Day 3 and now again
today), we still don't have an explanation. Not a guess we're confident in.
Not even a strong hypothesis. Just: it happens sometimes, and we don't know
why.

The instinct in a moment like that is to either panic about the deadline or
just let it run and hope. We did neither today. Instead we stopped
deliberately -- not because anything broke, but because it was the right
moment to ask "wait, are we actually on track, or have we just been busy?"
-- and the honest answer was: busy, yes. On track for everything the
reviewers asked for, not yet.

That's worth sitting with for a second. The multi-model matrix -- the part
that's eaten most of six days of laptop time, most of the debugging, most
of the drama with stale checkpoints and eval-set inconsistencies and a
SmoothQuant implementation that turned out to be subtly wrong on two
different architectures -- answers exactly one of the six things the ACL
reviewers actually asked for. One. The QLoRA comparison, the downstream
task evaluation, the real statistical test -- the things that would make
this paper's revision actually land differently with a reviewer -- haven't
been started. Not "in progress." Not started.

That's not a failure. It's just where we actually are, and it's better to
know that clearly today than to discover it the week before a deadline.

The good news, and it is real good news: today's five-hour mystery run cost
us *nothing* when we stopped it. That's the payoff of Wednesday's work
building the mid-training checkpointing -- the thing we built specifically
because we'd lost 2.5 hours of Qwen training to an identical situation and
it stung enough to fix properly. We asked, before building it, whether
pausing and resuming a training run could quietly hurt the model's quality
-- a real question, not a rhetorical one -- worked out the answer
(no, provided you save the right state), built it, and then tested it
against the actual way we stop things (a hard kill, not a polite Ctrl+C)
rather than trusting that it would probably work. It's now been used for
real, twice, across a multi-day gap, and it did exactly what it was supposed
to. That's a genuinely good feeling -- not "we ran an experiment," but "we
built a piece of infrastructure because we got burned, and now getting
burned that way again is impossible."

So today ends with an open question instead of a finished task, and that
feels like the right note to end on. The mystery slowdown is still a
mystery. The three unstarted reviewer critiques are still unstarted. But we
know exactly what we don't know now, which is a different and better
position than not knowing that we didn't know it.

Next time we sit down with this, the real decision isn't "keep running
Qwen until it's done." It's whether to keep going model-by-model in strict
sequence, or start the QLoRA and downstream-eval work now, in parallel,
given how unpredictable Qwen's timing has turned out to be. That's a
decision for next time, made with clear eyes instead of momentum.

---

## 2026-08-30

The mystery from the previous entry got solved, for real, not just
worked around. The five-hour-plus stuck-looking eval turned out to be
exactly what it looked like from the outside but couldn't be proven from
the outside: the process was genuinely exceeding the 6 GB card and Windows
was quietly falling back to system RAM, which explains everything --
high GPU utilization, no crash, no error, just glacial progress with no way
to distinguish it from a stuck loop except by actually measuring VRAM
directly, which is what finally happened. The fix (`EVAL_BATCH_SIZE=1`) took
eval from ~7 hours down to ~80 seconds in the test that verified it. That
felt like a real, clean win -- the kind where you can point at a before/after
number and know you actually fixed the thing rather than just moved it.

Decided to continue running the resumed matrix without interruption, and
to stop asking "how's it going" every few minutes -- both a discipline
thing (routine step-time noise isn't news) and a trust thing (the
infrastructure built over the last three days -- checkpointing, resumable
eval, per-step visibility -- means a bad outcome is recoverable, so there's
less need to hover).

## 2026-08-31

The "real, clean win" from yesterday turned out to be real but not
complete. Watching the next two control-seed runs finish, eval took ~27
minutes and then ~62 minutes -- nowhere near the ~80 seconds the fix was
supposed to guarantee. The instinct here could easily have been to shrug
and say "well, it's not the 7-hour disaster anymore, good enough" -- but
the honest version of "is this method fine, is anything lacking" (a
question asked back on Day 2, and still the right one to keep asking) meant
actually finding out why, not just noticing it was better than before.

Two real bugs were sitting in code that had already been trusted: eval was
quietly building a full backward-pass graph it never used, on every batch,
for the entire life of this project, and the training process's leftover
optimizer state was never cleared before eval started borrowing the same
6 GB. Neither is dramatic on its own. Together, on a card with this little
headroom, they were enough to erase most of yesterday's fix. It's a good
reminder that "verified with a clean benchmark" and "verified in
production" are different claims, and the gap between them is exactly
where a card this small has no slack to hide it.

Also asked directly, separately from the eval bug: is something in the
*training* loop's code -- not just VRAM -- causing the noisy step times?
Went and actually checked (data loading, error-tracking overhead,
checkpoint I/O) instead of re-asserting the existing VRAM story by default.
Nothing new turned up. That's a less satisfying entry than "found and fixed
a bug," but it's the honest one, and it's a different, more earned kind of
confidence in the VRAM explanation than just repeating it would have been.

Stopped for the day at a genuinely clean boundary -- not mid-run, not a
kill, but right after a seed's full training-eval-checkpoint cycle
finished -- after roughly eighteen hours of near-continuous runtime on this
one matrix. Three of Qwen's twelve QAT runs are done. The two eval bugs
found today can't help anything that already ran; they'll only prove
themselves the next time this is picked back up, on the very first fresh
run. That's next time's first thing to check, not an assumption to carry
in.

## 2026-09-06 / 2026-09-07

The very first prediction from last time's entry came true almost
immediately: the first QAT-strategy run exposed a bug none of the control
runs ever could have. One step took fifty-four minutes. The instinct to
just chalk it up to "the VRAM thing again" was strong -- it's the
explanation that's been right so many times this project -- but the
magnitude was wrong for that story (an order of magnitude past the worst
VRAM noise ever seen), and this time there was a real, checkable
alternative: quantization-error tracking, which only turns on for a real
QAT strategy, never for the control. Went and read the code instead of
reaching for the familiar answer, and found it: seventeen hundred syncs a
step, hiding in a feature that only existed to feed a different part of
the paper's mathematical section.

This session also had two smaller, quieter lessons. First: got asked to
stop the GPU immediately, mid-run, for something unrelated -- and it was
fine, because three days ago a version of this exact question was worked
through carefully rather than assumed. Second: a fix for one bug introduced
a new one in the same afternoon (a stale `.item()` call that assumed the
old code path was still there), and it surfaced as an actual crash a few
hours later. Caught it, fixed it, and it cost real but bounded time -- not
a full retraining, just a resume. That's the whole point of building
checkpointing this carefully: mistakes made *while building the
infrastructure* get to be cheap too, not just mistakes in the experiment
itself.

The genuinely open question from today isn't a bug -- it's a result.
`weights_only` behaved almost identically to its control on seed 42, and
dramatically worse than its control on seed 1337. That's a bigger seed-to-
seed swing than this exact configuration ever showed on either of the
other two models. It might be the most interesting finding of the whole
Qwen matrix, or it might be an artifact of something not yet understood
about that specific run. Resisting the urge to explain it before there's a
third seed to look at.

## 2026-09-09

`weights_only`'s third seed came in and quietly resolved yesterday's open
question -- 18.23, close to seed 42's 18.69, which makes seed 1337 the
outlier rather than the strategy being broadly unstable. That felt like a
clean, satisfying close to a loose thread. It was also the last clean,
satisfying thing that happened today.

`activations_only` started, and within the first run it was obvious
something different was happening -- not the familiar VRAM noise, a real
loss spike, 43x in a single step. Checked the obvious alternative
explanation (VRAM) and ruled it out directly rather than assuming. Then
it happened again, bigger. Then a second seed showed the same pattern,
tighter and more frequent. Then a third seed took that pattern and didn't
stop -- watched it, step by step, as the loss went from ordinary to
hundreds to millions to a number with eighteen digits before the
floating-point math simply gave up and returned NaN. That's not a
metaphor for "the training got worse." It's a literal, traceable arithmetic
event, and it was satisfying in a strange way to watch it happen in real
time and understand exactly why, rather than just seeing a ruined final
number days later with no idea how it got there.

The question that came right after wasn't technical, it was about time
and ownership: keep watching four more hours of a number that was already
known to be NaN, or ask. Asked. Waited for the answer without touching
anything, which meant the run kept going and finished exactly as
predicted -- NaN from step 287 all the way to 500. Nothing was lost by
waiting; if anything, watching it run to the end removed any doubt about
whether it might have recovered.

The fix itself is almost anticlimactic after a day like this: gradient
clipping, a completely standard technique that's honestly more surprising
in its absence than its presence would be in any other training setup.
But the harder, more honest part of today wasn't writing that one line --
it was deciding what to do with the data that already existed. `none` and
`weights_only` never showed a single spike; leaving those alone and
redoing only what actually broke, including the one `activations_only`
seed that *didn't* fully diverge, felt like the right kind of
carefulness -- not redoing everything reflexively, and not leaving a
mismatched half-fixed dataset behind either.

Whether clipping actually solves this or just moves the divergence further
out is genuinely unknown right now, and that's fine to end the day on. Six
runs are queued for next time, with a very specific thing to check first:
does the `[grad clip]` log line show up early and often on `activations_only`,
the way today's uncontrolled logs would have if it had existed then.

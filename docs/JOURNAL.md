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

## 2026-09-13

The clipping fix got its first real test today, and it held. Step 1
of the redo needed clipping (a gradient norm 121 times over threshold),
and the run just... kept going. A hundred and fifteen steps in, still no
spike, still no NaN, still a boring normal loss number -- which, after
watching that number climb to eighteen digits four days ago, is a genuinely
satisfying kind of boring.

The more interesting moment today wasn't the fix working, it was catching
my own sloppy read of *how* it was working. I'd been reporting the clip
count per ten-minute window and said out loud that it looked like it was
"tapering off" -- fewer clips in later windows than the first one. The
user asked a simple, direct question back: "so all of them are clipping?"
That's the kind of question that only gets asked when someone is actually
tracking the numbers rather than trusting the summary, and it was the
right instinct -- checking properly showed every single window had the
same ratio as the first one. Nothing had tapered. I'd been comparing raw
counts across windows of different lengths without noticing the windows
themselves were the same length the whole time, which made a flat 100%
line look like a downward slope.

That's worth sitting with for a second, the same way the original
divergence was. It wasn't a big mistake, and it didn't change any
decision -- the fix is still working regardless of whether the clip rate
is 100% or 60%. But it's exactly the kind of small, confident,
unverified claim that this project has tried hard not to let slide,
going all the way back to the first session's "we didn't crash, so it's
probably fine." Getting asked a plain question and having to actually go
check, rather than getting to just restate the summary more confidently,
is the mechanism that catches this -- not some higher level of personal
care the second time around.

## 2026-09-17

Eleven of the twelve Qwen runs are in, and the fix held for all of them --
`activations_only`'s redo finished all three seeds clean, and `both`'s
first two seeds came in clean as well, no spikes, no NaN, nothing that
needed a second look. That's the payoff of Day 8 and 9 actually showing up
in the data rather than just in the log lines: six runs went by this
session without a single moment of "wait, is this okay?"

The twelfth run -- the very last one needed to close this matrix -- got
stopped on purpose, mid-training, at step 250 of 500, because it was time
to stop for the day. That's a completely ordinary thing to do at this
point in the project, and it was worth actually checking rather than
trusting on faith: this stop was a hard kill, not the polite signal the
checkpointing code is built to catch, and the honest question was whether
that mattered. It didn't turn out to, but "didn't turn out to" is only a
real answer once you've loaded the checkpoint and read the step number
back out, which is what happened -- 250, three steps behind where the
process actually was when it died. Three steps is nothing. But the whole
point of building resumability this carefully, several sessions ago, was
so that a question like this could be answered with a number instead of a
shrug.

So the matrix ends today at 11 of 12, one clean resume away from done.
That's a good, clear place to stop -- closer to finished than confused,
and with the next action (resume, finish, then actually sit down and
decide what comes after Section A) obvious rather than improvised.

## 2026-09-18

Twelve of twelve. The resume from step 250 picked up exactly where the
checkpoint said it would, the training half was noticeably faster than the
day before for reasons that turned out to be worth actually chasing down
(a system reboot, and possibly the accumulated weight of over a dozen
background monitoring loops that never got cleanly torn down between
sessions) rather than shrugged off as "computers are like that
sometimes" -- and the final eval landed at PPL=33.55, in the same healthy
range as its two sibling seeds. Section A, the thing this whole
multi-week matrix has been building toward, is done.

Then came the part that felt different from every other day of this
project: turning all of it into something that reads like a paper instead
of a status table. Sitting down to actually populate the outline with the
real numbers surfaced two things that had been sitting quietly in the
data the whole time without anyone noticing, because nobody had put every
model's numbers next to each other in the same table before. Qwen's
weight/activation error ratio is the *lowest* of the three models, not the
highest -- which is backwards from what the fused-QKV story would predict
if it were the whole explanation, since Qwen doesn't even have fused QKV.
And the seed=1337 weights-only outlier that got flagged back on Day 7 and
then quietly not revisited is still just sitting there, unexplained, now
load-bearing enough in the mean that it can't be glossed over in a paper
the way it could be glossed over in a status update.

Neither of those is a comfortable thing to find while assembling a draft
that's supposed to be making the project's case. The easy version of this
moment would have been to average past both of them, report the clean
headline number, and let the draft look more finished than it is. That's
exactly the instinct this project has been trying to build a habit of
resisting since the very first session's "we didn't crash, so it's
probably fine" -- and it held here too. Both wrinkles are in the draft,
named as open questions, not smoothed into the prose.

It's a strange kind of milestone: the data collection phase that ate most
of this project's calendar time is actually finished, and the honest
answer to "is the paper done" is still no, more clearly and specifically
no than it was yesterday, because now there's an actual document that
says exactly what's missing instead of a table that could be read either
way. That feels like real progress, not a consolation prize for not being
finished.

## 2026-09-18/19 (Day 12/13)

Given an order -- D, then C, then B -- and for once the order mattered
less than what showed up while doing the work. Every single one of the
three sections turned up something that wasn't in the plan.

D was supposed to be the boring one: backfill some data, run a test that
was already written, report the p-values. Instead the backfill's own
sanity check caught a real bug -- the final checkpoints had been rounding
a quantization scale to half precision the whole time, and Pythia's
activation-heavy configs were quietly understating their own damage by
up to 2.75% because of it. The instinct to wave that off as "close
enough, under the noise floor for most of these" was right there, and it
would have been wrong to take it: the fact that OPT showed the exact same
bias at exactly the scale its own architecture would predict is what
turned a nuisance into a second piece of evidence for a mechanism this
project already believed in. Small bugs are still worth chasing when they
happen to land exactly where your own theory says they should.

C was supposed to just confirm what perplexity already said, and mostly
it did -- Pythia's activation-QAT damage doesn't just move a perplexity
number, it nearly halves next-word accuracy outright, which is a much
more visceral way to see the same finding. But it also handed back
something perplexity had been hiding: Qwen's weight-only quantization,
which looked cheap in every perplexity table so far, costs real
downstream accuracy across all three seeds. That's not the seed=1337
outlier showing up again in a new costume -- it's consistent, it's new,
and it means "weight quantization is basically free" was true for exactly
two of the three models tested, not all three.

B was the one that almost went wrong twice in the span of an hour. First,
a smoke test that looked completely dead -- zero output, three minutes of
silence -- got killed as a hang before it was actually just Windows fully
buffering a piped process's stdout, the exact bug this project fixed once
already on Day 5 and then never carried over to any script written since.
Then, right after, a genuinely finished training run's eval turned out to
be scoring itself on the same data it had just trained on, at full size,
not the fixed held-out slice every other number in this paper uses --
caught with the eval already more than half done, and worth throwing away
and rerunning correctly rather than keeping a number that wouldn't have
meant what it looked like it meant. Both mistakes were mine, both were
caught before they cost anything real, and the second one in particular
is a quieter version of the exact confound this project was built to
avoid in the first place -- it's apparently possible to reintroduce that
lesson by accident even after learning it once.

The actual science landed clean, in the end: the frontier model doesn't
fit under full QAT on this card -- not cleanly, it hits the same silent
oversubscription slowdown as Qwen once did, which is itself a second
confirmation of a pattern instead of a one-off -- and QLoRA fits with
room to spare. Three sections done, three real findings that weren't on
the agenda when the day started, and two mistakes caught in the same
session they were made rather than discovered later by someone reading
the paper more carefully than I did. That's a good trade.

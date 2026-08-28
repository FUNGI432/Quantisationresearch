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

# A video, stated

A tower that reads a video states how (`jlm.Video`), and the engine reads
nothing else: no sampler, prompt or layout of its own per processor. The
statement grew a flag per processor instead -- Joined, PerSecond and
EveryFrame for Qwen2-VL, Steps, MaxSeconds, StampFirst and StampLast for
GLM-4V, Whole, StampIndex and Between for InternVL -- and `model.VideoPlan`
switched on them. A field with one family behind it is that family's code
(AGENTS.md, "the architecture is the converter's alone"). This is the
statement that replaces them; the converter writes each family's processor
into it, the engine evaluates it.

## What it states

    DefaultFps   the rate a request that states none is read at
    Count        how many frames are sampled: an expression (jlm.Expr) of
                 the video's frames N and rate R
    Index        sampled frame i's index: an expression of N, R, the count
                 K and i, truncated toward zero and held to [0, N-1]
    Once         a frame the same as the one sampled before it is dropped
    MaxFrames    the most frames Count gives: what the entry's taps are
                 sized for
    Group        frames one picture of the tower reads: 2 for a pair
                 entry, 1 for a frame read whole; an odd count's last frame
                 repeated to fill the last group
    Pre, Entry   the frames' own resize and entry; no entry stated: each
                 frame is read as a picture is, through the picture's
                 entry, at its last square -- under the picture's steps, or
                 under Pre where stated, the picture's steps with one
                 resize of its own (jlm.Video.PicturesValid)
    Stride       where stated, the groups are one grid, group j at j times
                 this on the text model's temporal axis (an expression of
                 R and the group's frames); where not, each group is a
                 picture span of its own
    Prompt       pieces, in order

A piece is one of

    ids          token ids, as stated (a marker the tokenizer would not
                 make of text)
    text         text, encoded with the text beside it
    value        an expression's value, written by a format
    groups       its body once per group
    span         the group's rows, between the tower span's markers

and a format is one of

    decimal      to D places, rounded correctly or truncated
    clock        a pattern of hours, minutes and seconds, each zero-padded
                 to a stated width or not
    table        strings[int(value)], the strings the converter wrote

The expressions read, by operand: N, R, K, i for the sampler; N, R, K, the
group j, its first frame's index F0 and seconds T0, its last frame's index
F1 and seconds T1, and the video's seconds N / R for the prompt and the
stride.

Text at either end of the prompt is the video's lead and tail
(`model.VideoParts`): the chat (`Model.ChatSpansMedia`, which the server
and `jitllm run` build their prompts with) encodes them with the
template's text beside them, as a processor expands its placeholder into a
string before tokenizing it. `Model.VideoSpans`, for a caller placing a
video with no text beside it, encodes them alone.

## The processors, as statements

    Qwen3-VL     Count min(max(int(N/R*2), 4), 768, N); Index
                 roundEven(i*((N-1)/(K-1))), N-1 at the last; pairs;
                 groups: "<", decimal T0..T1 mean to 1 place, " seconds>",
                 span
    Qwen2-VL     Count N; Index i; pairs; Stride 1 (Qwen2.5-VL:
                 2*int(2/R)); prompt: the span markers around the groups
    GLM-4V       Count and Index over d = roundEven((N-1)/R)+1 seconds,
                 whole steps of R/2 up to 300 seconds, else a linspace in
                 seconds; Once; pairs; groups: span, then decimal T0
                 truncated to whole seconds
    InternVL     the picture's sampler; frames whole; groups: "Frame",
                 decimal j+1, ": ", span, with "\n" between
    Gemma 4      Count min(N, 32); Index f32(i*(N/K)); frames whole, its
                 Pre the picture's budgeted resize at 70 soft tokens;
                 groups: clock T0 as MM:SS, " ", span, " " between
    SmolVLM      Count max(min(roundEven(N/R), 64), 1); Index a linspace
                 a second in from either end past 66 seconds, truncated;
                 Once; frames whole; text with a table of the counts in
                 words and the duration as a clock H:MM:SS; groups:
                 "\nFrame from ", clock T0 as MM:SS, ":", span; "\n\n"

## The expression language

Count, Index and Stride need floor, ceil, truncation, rounding half to even
and a rounding to binary32 (Gemma 4's torch.arange in double cast to float):
generic functions of `jlm.Fn`, lowered by the composer on every device as
the other rounding is. The sampler's expressions are evaluated on the host
in float64, as numpy and torch compute them: geometry from the sizes, as the
picture's resize plan is.

## Not yet stated

The frames' resize steps (`jlm.PreVideoResize`, Qwen's video smart_resize)
and several of the picture's (`PreSmartResize`, `PreNaFlex`,
`PreLongBox`) are still one processor's resize each. They are the next
statement to generalise: a resize whose output size is an expression of the
picture's, with the processors' searches (NaFlex's scale search) stated as
bounded iterations.

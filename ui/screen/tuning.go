package screen

import (
	"fmt"
	"strconv"

	"github.com/gogpu/ui/core/scrollview"
	"github.com/gogpu/ui/core/slider"
	"github.com/gogpu/ui/core/textfield"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/ui/app"
)

// Tuning is the sampling and context panel, shown beside the conversation.
// Sampling applies to the next reply and the context size to the next load,
// and the panel says which: the KV window is allocated when the session is
// built, so it cannot move under a live model.
func Tuning(sh *app.Shell) widget.Widget {
	body := primitives.VBox(
		section(sh, "Sampling", sampling(sh)),
		section(sh, "Context", context(sh)),
	).Gap(sh.P.Space.S).Padding(sh.P.Space.M)

	return primitives.Box(scrollview.New(body, scrollview.PainterOpt(sh.P.Scrollbar))).
		Background(sh.P.Surface())
}

func sampling(sh *app.Shell) widget.Widget {
	// The knobs are [session.SamplingKnobs()], the terminal's too, in the
	// sampler's order; their comment says why that order.
	rows := make([]widget.Widget, 0, len(session.SamplingKnobs())+2)
	for _, k := range session.SamplingKnobs() {
		rows = append(rows, knob(sh, k.Name, k.Min, k.Max, k.Step, k.Get, k.Set, k.Caption))
	}
	rows = append(rows, seedField(sh),
		note(sh, "Applied in llama.cpp's order, so a prompt samples the same in both."))

	return primitives.VBox(rows...).Gap(2)
}

// seedField is a text field because a seed is a value, not a range.
func seedField(sh *app.Shell) widget.Widget {
	txt := state.NewSignal(strconv.FormatInt(sh.Store.Sampling.Get().Seed, 10))
	return primitives.VBox(
		sh.P.Role(primitives.Text("Seed").Color(sh.P.Muted()), sh.P.Type.LabelMedium),
		primitives.Box(textfield.New(
			textfield.ValueSignal(txt),
			textfield.Placeholder("0"),
			textfield.OnChange(func(s string) {
				n, err := strconv.ParseInt(s, 10, 64)
				if err != nil && s != "" {
					return
				}
				set(sh, func(v *app.Sampling) { v.Seed = n })
			}),
			textfield.PainterOpt(sh.P.TextField),
		)).Width(200),
	).Gap(sh.P.Space.XS).PaddingXY(0, sh.P.Space.XS)
}

func context(sh *app.Shell) widget.Widget {
	val := state.NewSignal(float32(sh.Store.MaxSeq.Get()))
	state.NewComputed(func() int {
		n := int(val.Get())
		if n != sh.Store.MaxSeq.Get() {
			sh.Store.MaxSeq.Set(n)
			sh.Cfg.MaxSeq = n
		}
		return n
	}, val.AsReadonly())

	caption := reactive(sh, sh.Store.MaxSeq.AsReadonly(), func() string {
		return fmt.Sprintf("%d tokens", sh.Store.MaxSeq.Get())
	})

	// The caption shrinks, not the label: Expanded takes the leftover, so an
	// expanded label beside a long caption got no width at all.
	return primitives.VBox(
		primitives.HBox(
			sh.P.Role(primitives.Text("Context window").Color(sh.P.Muted()), sh.P.Type.LabelMedium),
			primitives.Expanded(line(sh, caption).Align(widget.TextAlignRight)),
		).Gap(8),
		slider.New(
			slider.ValueSignal(val),
			slider.Min(512), slider.Max(32768), slider.Step(512),
			slider.PainterOpt(sh.P.Slider),
		),
		// The model's trained context is a ceiling too: more allocates a cache it
		// cannot use coherently, and the header reports the clamp at load. The
		// window is allocated up front (maxseq x kvdim x 4 x 2 x layers), so a
		// smaller one is the difference between fitting and paging.
		note(sh, "Takes effect on the next load, up to the model's own limit. A smaller window uses less memory."),
	).Gap(4)
}

// knob is one labelled slider over a field of Sampling.
func knob(
	sh *app.Shell, label string, min, max, step float32,
	get func(app.Sampling) float32,
	put func(*app.Sampling, float32),
	caption func(app.Sampling) string,
) widget.Widget {
	// The slider owns its signal and writes through on change: a
	// Signal[Sampling] is replaced wholesale on every Set, so sliders bound to
	// its fields would clobber each other.
	val := state.NewSignal(get(sh.Store.Sampling.Get()))
	cap := reactive(sh, sh.Store.Sampling.AsReadonly(), func() string {
		return caption(sh.Store.Sampling.Get())
	})
	return primitives.VBox(
		primitives.HBox(
			// The caption shrinks, not the label; see above.
			sh.P.Role(primitives.Text(label).Color(sh.P.Muted()), sh.P.Type.LabelMedium),
			primitives.Expanded(line(sh, cap).Align(widget.TextAlignRight)),
		).Gap(8),
		slider.New(
			slider.ValueSignal(val),
			slider.Min(min), slider.Max(max), slider.Step(step),
			slider.OnChange(func(v float32) { set(sh, func(s *app.Sampling) { put(s, v) }) }),
			slider.PainterOpt(sh.P.Slider),
		),
	).Gap(sh.P.Space.XS).PaddingXY(0, sh.P.Space.XS)
}

// set edits one field of the sampling settings and republishes the whole
// struct, which is how a Signal over a value type has to be written.
func set(sh *app.Shell, edit func(*app.Sampling)) {
	s := sh.Store.Sampling.Get()
	edit(&s)
	sh.Store.Sampling.Set(s)
	sh.Cfg.Sampling = s
}

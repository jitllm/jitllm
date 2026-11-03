package screen

import (
	"github.com/gogpu/ui/core/button"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/widgets"
)

// sessionProblem is the error banner under the header: what went wrong, why,
// and the fix as a button. It takes no space while there is no problem. It is
// a banner rather than a dialog because core/dialog's fixed height cut off the
// end of the message, where the fix is (UPSTREAM.md #13).
func sessionProblem(sh *app.Shell) widget.Widget {
	st := sh.Store
	field := func(f func(app.Problem) string) state.ReadonlySignal[string] {
		return state.NewComputed(func() string { return f(st.Problem.Get()) }, st.Problem.AsReadonly())
	}

	title := widgets.NewParagraph("").
		ContentSignal(field(func(p app.Problem) string { return p.Title })).
		FontSize(sh.P.Type.TitleSmall.FontSize).Bold().
		Color(sh.P.Colors.OnErrorContainer).Font(sh.P.Prose())
	detail := widgets.NewParagraph("").
		ContentSignal(field(func(p app.Problem) string { return p.Detail })).
		FontSize(sh.P.Type.BodyMedium.FontSize).
		Color(sh.P.Colors.OnErrorContainer).MaxLines(6).Font(sh.P.Prose())

	act := widgets.Hide(
		state.NewComputed(func() bool { return st.Problem.Get().Action != "" }, st.Problem.AsReadonly()),
		button.New(
			button.TextReadonlySignal(field(func(p app.Problem) string { return p.Action })),
			button.SizeOpt(button.Small),
			button.VariantOpt(button.Filled),
			button.OnClick(func() { sessionProblemFix(st) }),
			button.PainterOpt(sh.P.Button),
		),
	)
	dismiss := button.New(
		button.TextOpt("Dismiss"),
		button.SizeOpt(button.Small),
		button.VariantOpt(button.TextOnly),
		button.OnClick(func() { st.Problem.Set(app.Problem{}) }),
		button.PainterOpt(sh.P.Button),
	)

	return widgets.Hide(
		state.NewComputed(func() bool { return st.Problem.Get().Seq != 0 }, st.Problem.AsReadonly()),
		primitives.Box(
			primitives.VBox(title, detail, primitives.HBox(act, dismiss).Gap(sh.P.Space.S)).
				Gap(sh.P.Space.XS).
				CrossAlign(primitives.CrossAxisStretch),
		).
			Padding(sh.P.Space.M).
			Background(sh.P.Colors.ErrorContainer),
	)
}

// sessionProblemFix is the banner's button: it takes the banner down, then
// runs the fix -- in that order, because the fix may report a problem of its
// own.
func sessionProblemFix(st *app.Store) {
	p := st.Problem.Get()
	st.Problem.Set(app.Problem{})
	if p.Do != nil {
		p.Do()
	}
}

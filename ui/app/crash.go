package app

import (
	"github.com/gogpu/ui/core/button"
	"github.com/gogpu/ui/core/scrollview"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/common/crash"
	"github.com/jitllm/jitllm/ui/widgets"
)

// ShowCrash puts a crash report over every screen. Safe from any goroutine:
// the report is published on the UI goroutine, like a dialog.
func (s *Shell) ShowCrash(r crash.Report) {
	s.Post(func() { s.Store.Crash.Set(r) })
}

// crashActions are what the crash panel's buttons do, kept apart so a test
// reaches them without a window.
func (s *Shell) copyCrash() {
	r := s.Store.Crash.Get()
	s.Copy(r.Text)
	s.SetStatus("crash report copied")
}

func (s *Shell) fileCrash() {
	r := s.Store.Crash.Get()
	// Copied first: a long report's link carries a summary and asks for the
	// paste.
	s.Copy(r.Text)
	if err := crash.OpenURL(crash.IssueURL(r)); err != nil {
		s.SetStatus("could not open the browser: " + err.Error())
	}
}

// crashPanel is the crash report: what happened, the whole report (selectable,
// scrolled), and Copy, Open an issue and Dismiss. A panel rather than a
// dialog because core/dialog has room for two lines (UPSTREAM.md #13) and the
// report is a stack. It takes no space while there is no report.
func (s *Shell) crashPanel() widget.Widget {
	st := s.Store
	field := func(f func(crash.Report) string) state.ReadonlySignal[string] {
		return state.NewComputed(func() string { return f(st.Crash.Get()) }, st.Crash.AsReadonly())
	}
	c := s.P.Colors

	head := widgets.NewParagraph("").
		ContentSignal(field(func(r crash.Report) string {
			return "jitllm crashed. The report below has no prompt or chat in it; copy it into an issue so it can be fixed."
		})).
		FontSize(s.P.Type.TitleSmall.FontSize).Bold().
		Color(c.OnErrorContainer).Font(s.P.Prose())
	where := widgets.NewParagraph("").
		ContentSignal(field(func(r crash.Report) string {
			if r.Path == "" {
				return r.Title
			}
			return r.Title + "\nsaved to " + r.Path
		})).
		FontSize(s.P.Type.BodySmall.FontSize).
		Color(c.OnErrorContainer).Font(s.P.Prose())
	text := widgets.NewParagraph("").
		ContentSignal(field(func(r crash.Report) string { return r.Text })).
		FontSize(s.P.Type.BodySmall.FontSize).
		Color(c.OnSurface).Font(s.P.Mono()).Selectable()

	btn := func(label string, v button.Variant, fn func()) widget.Widget {
		return button.New(
			button.TextOpt(label),
			button.SizeOpt(button.Small),
			button.VariantOpt(v),
			button.OnClick(fn),
			button.PainterOpt(s.P.Button),
		)
	}
	actions := primitives.HBox(
		btn("Copy report", button.Filled, s.copyCrash),
		btn("Open an issue", button.Outlined, s.fileCrash),
		btn("Dismiss", button.TextOnly, func() { st.Crash.Set(crash.Report{}) }),
	).Gap(s.P.Space.S)

	return widgets.Hide(
		state.NewComputed(func() bool { return st.Crash.Get().Text != "" }, st.Crash.AsReadonly()),
		primitives.Box(
			primitives.VBox(
				head, where,
				primitives.Box(scrollview.New(text, scrollview.PainterOpt(s.P.Scrollbar))).
					Height(320).Padding(s.P.Space.S).Background(c.Surface),
				actions,
			).Gap(s.P.Space.XS).CrossAlign(primitives.CrossAxisStretch),
		).
			Padding(s.P.Space.M).
			Background(c.ErrorContainer),
	)
}

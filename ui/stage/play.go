package stage

import (
	"time"

	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/mock"
	"github.com/jitllm/jitllm/ui/screen"
)

// Play runs sc in a fresh window built over package mock, and hands each
// frame to each, with the machine's fonts loaded when fonts is set: one named "start" before any step, then one per step. It
// returns every step that did not do what it says -- an Until that never came
// true, a Want that failed -- since a step that did nothing still renders a
// clean frame.
func Play(sc mock.Scenario, cfg *app.Config, w, h int, fonts bool, each func(step string, f Frame)) []string {
	var failed []string
	sh := app.NewShell(nil, cfg)
	if fonts {
		sh.Fonts.Load()
	}
	d, e := mock.Deps(sh)
	screen.Install(sh, d)
	sh.Store.Tab.Set(sc.Tab)
	st := New(sh, w, h)
	each("start", st.Frame())
	for _, s := range sc.Steps {
		if s.Do != nil {
			s.Do(sh, e)
		}
		f := st.Frame()
		for deadline := time.Now().Add(5 * time.Second); s.Until != nil && !s.Until(sh); {
			if time.Now().After(deadline) {
				failed = append(failed, s.Name+": what it waits for never happened")
				break
			}
			time.Sleep(10 * time.Millisecond)
			f = st.Frame()
		}
		if s.Want != nil {
			if why := s.Want(sh); why != "" {
				failed = append(failed, s.Name+": "+why)
			}
		}
		for _, want := range s.Shows {
			if !f.Paints(want) {
				failed = append(failed, s.Name+": the frame does not show "+want)
			}
		}
		each(s.Name, f)
	}
	return failed
}

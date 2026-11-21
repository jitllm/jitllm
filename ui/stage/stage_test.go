package stage_test

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/mock"
	"github.com/jitllm/jitllm/ui/stage"
)

// Every scenario, at both window sizes the shots use and in both themes, must
// render every frame without text painted over text, past its clip or past
// the window -- the way the app looks broken. The frames are the running app's
// order: built empty, then filled by the engine.
func TestEveryScenarioRendersClean(t *testing.T) {
	for _, sc := range mock.Scenarios() {
		for _, size := range [][2]int{{1440, 900}, {1000, 700}} {
			for _, dark := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%dx%d/dark=%v", sc.Name, size[0], size[1], dark), func(t *testing.T) {
					cfg := app.DefaultConfig(app.Env{})
					cfg.Light = !dark
					cfg.ModelDirs = []string{t.TempDir()}
					frames := 0
					failed := stage.Play(sc, cfg, size[0], size[1], false, func(step string, f stage.Frame) {
						frames++
						if len(f.Texts) == 0 {
							t.Errorf("%s: the frame painted no text at all", step)
						}
						for _, p := range f.Problems() {
							t.Errorf("%s: %s", step, p)
						}
					})
					for _, why := range failed {
						t.Error(why)
					}
					if frames != len(sc.Steps)+1 {
						t.Errorf("%d frame(s) for %d step(s)", frames, len(sc.Steps))
					}
				})
			}
		}
	}
}

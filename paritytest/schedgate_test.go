package paritytest

import (
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

// A schedule is the planner's to choose from a definition's shape and the
// device's facts (compose.Plan). A parity test that hands the composer a
// schedule, or a tier generator that builds one, has chosen main's kernel by
// hand. The count only falls; lower the ceiling with the change that lowers it.
const handSchedulesCeiling = 10

var handSchedule = regexp.MustCompile(`compose\.Schedule\{|\bPlanAt\(|\bemitAt\(t|\bformed\(t,|\.Emit\(compose\.Schedule`)

func TestNoHandPickedSchedules(t *testing.T) {
	var files []string
	for _, glob := range []string{"*_test.go", "../jit/gpu/tier/*.go"} {
		m, err := filepath.Glob(glob)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, m...)
	}
	n := 0
	for _, f := range files {
		if filepath.Base(f) == "schedgate_test.go" {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for _, loc := range handSchedule.FindAllIndex(b, -1) {
			n++
			t.Logf("%s: %s", f, b[loc[0]:loc[1]])
		}
	}
	t.Logf("hand-picked schedules: %d (ceiling %d)", n, handSchedulesCeiling)
	if n > handSchedulesCeiling {
		t.Errorf("hand-picked schedules rose to %d past %d: the planner chooses a schedule", n, handSchedulesCeiling)
	}
}

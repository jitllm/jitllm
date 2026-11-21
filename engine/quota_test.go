package engine

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
)

// TestAQuotaBudgetsTheFramesFromTheirFirstRead: an instance's page quota
// reaches its CPU device's frames before they read a page, a preload's among
// them -- a preload under a quota below the model reads nothing a run without
// one does not -- and each instance of one model on one CPU device pages
// through a view of its own at its own quota, re-budgeted alone as it
// changes, whatever the others' do.
func TestAQuotaBudgetsTheFramesFromTheirFirstRead(t *testing.T) {
	path := jlmOf(t, models()["stories15M"])
	probe, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	var all uint64
	n := probe.Container.NPages()
	for i := range n {
		all += probe.Container.PageBytes(i)
	}
	probe.Close()
	if n < 4 {
		t.Fatalf("%d pages: a quota below the model cannot be told from one above it", n)
	}

	reads := func(mo ...model.Option) int64 {
		m, err := model.Open(path, mo...)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		st := testInstance(t, m, 8, WithPageBudget(all/2))
		defer st.Close()
		if err := st.fr.WaitPreload(); err != nil {
			t.Fatal(err)
		}
		if b := st.fr.Budget(); b == 0 || b >= all {
			t.Fatalf("the frames' budget is %d of %d page bytes: the quota did not reach them", b, all)
		}
		return m.PagerReads()
	}
	if pre, cold := reads(model.WithPreload(4)), reads(); pre != cold {
		t.Fatalf("a preload under a quota below the model issued %d reads against %d without one: "+
			"it read before the quota reached the frames", pre, cold)
	}
	// The budget is the frames' as they are made, not set after: a preload
	// the making starts would otherwise race it. Read inside the maker the
	// CPU device calls, where a preload's first check would read it too.
	{
		m, err := model.Open(path, model.WithPreload(4))
		if err != nil {
			t.Fatal(err)
		}
		cd, err := OpenCPU()
		if err != nil {
			t.Fatal(err)
		}
		q := cd.NewQuota(m.Id, all/2)
		var made uint64
		if _, err := cd.Frames(m.Id, func(o ...jlm.FramesOption) (*jlm.Frames, error) {
			fr, err := m.NewFrames(o...)
			if err == nil {
				made = fr.Budget()
			}
			return fr, err
		}); err != nil {
			t.Fatal(err)
		}
		q.Drop()
		cd.Close()
		m.Close()
		if made != all/2 {
			t.Fatalf("the frames were made with a budget of %d, want the quota's %d: a preload "+
				"their making starts reads past the quota until it lands", made, all/2)
		}
	}

	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	cpu := sharedCPU(t)
	a := testInstance(t, m, 8, cpu, WithPageBudget(all/4))
	defer a.Close()
	b := testInstance(t, m, 8, cpu, WithPageBudget(all/2))
	if a.fr == b.fr {
		t.Fatal("two instances on one CPU device page through one view of the frames: one order and one budget for both")
	}
	if ga, gb := a.fr.Budget(), b.fr.Budget(); ga != all/4 || gb != all/2 {
		t.Fatalf("quotas of %d and %d keep %d and %d, want each its own", all/4, all/2, ga, gb)
	}
	b.Close()
	if got := a.fr.Budget(); got != all/4 {
		t.Fatalf("the other instance closed and this one's view keeps %d, want its %d", got, all/4)
	}
	a.SetPageBudget(all / 3)
	if got := a.fr.Budget(); got != all/3 {
		t.Fatalf("a quota set to %d keeps %d", all/3, got)
	}
}

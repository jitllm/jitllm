package engine

import (
	"slices"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// transitionModels are the step 6 gates' models: a dense one, and a mixture
// whose experts are pages of their own.
func transitionModels() []string { return []string{"stories15M", "qwen3moe"} }

// teacherForced is ids run through in one token at a time, every logit row.
func teacherForced(t *testing.T, in *Instance, ids []int32) [][]float32 {
	t.Helper()
	out := make([][]float32, 0, len(ids))
	for _, id := range ids {
		lg, err := in.Forward(id)
		if err != nil {
			t.Error(err)
			return nil
		}
		out = append(out, slices.Clone(lg))
	}
	return out
}

// sameRows reports the first row of got that is not want's, bit for bit, -1
// for none.
func sameRows(got, want [][]float32) int {
	if len(got) != len(want) {
		return 0
	}
	for i := range want {
		if !slices.Equal(got[i], want[i]) {
			return i
		}
	}
	return -1
}

// TestTwoInstancesReadAPageOnce is instance paging's held-once transition
// (docs/design/instance-paging.md step 6): two instances of one model on one
// CPU device decode at once -- their scans overlapping -- each as an instance
// alone does, bit for bit, and between them read every page once: the
// file's page-ins are one decode's. Two instances on two CPU devices read
// every page twice, which the gate sees.
func TestTwoInstancesReadAPageOnce(t *testing.T) {
	for _, name := range transitionModels() {
		t.Run(name, func(t *testing.T) {
			path := jlmOf(t, models()[name])
			ids := []int32{1, 7, 42, 3, 99, 5, 17, 2}
			faults := func(shared bool) (int64, [2][][]float32) {
				m, err := model.Open(path)
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				a, b := []Option{}, []Option{}
				if shared {
					cpu := sharedCPU(t)
					a, b = append(a, cpu), append(b, cpu)
				}
				x := testInstance(t, m, len(ids)+1, a...)
				defer x.Close()
				y := testInstance(t, m, len(ids)+1, b...)
				defer y.Close()
				var got [2][][]float32
				var wg sync.WaitGroup
				for i, in := range []*Instance{x, y} {
					wg.Add(1)
					go func() { defer wg.Done(); got[i] = teacherForced(t, in, ids) }()
				}
				wg.Wait()
				return m.PageFaults(), got
			}
			m, err := model.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			solo := testInstance(t, m, len(ids)+1)
			want := teacherForced(t, solo, ids)
			solo.Close()
			once := m.PageFaults()
			m.Close()
			if once == 0 {
				t.Fatal("a decode read no page: this gate proved nothing")
			}

			n, got := faults(true)
			for i := range got {
				if at := sameRows(got[i], want); at >= 0 {
					t.Errorf("instance %d beside the other parts from one alone at position %d", i, at)
				}
			}
			if n != once {
				t.Errorf("two instances on one CPU device read %d pages, one alone %d: a page both run was read twice", n, once)
			}
			if twice, _ := faults(false); twice < 2*once {
				t.Errorf("two instances on two CPU devices read %d pages against %d once: the gate cannot tell held twice from held once", twice, once)
			}
			t.Logf("one decode reads %d pages; two instances on one CPU device read %d", once, n)
		})
	}
}

// TestAnInstanceOnTheCPUAndOneOnACardShareTheFrames is instance paging's
// first transition (docs/design/instance-paging.md step 6): two instances
// of one model on one CPU device -- one running every block on the CPU under
// a quota of half its pages, the other with half its blocks on a card --
// decode at once, each as the same instance alone does, bit for bit, from
// frames they share within the quota; then the CPU's closes and the card's
// decodes on as alone, the frames still its own.
func TestAnInstanceOnTheCPUAndOneOnACardShareTheFrames(t *testing.T) {
	for _, name := range transitionModels() {
		t.Run(name, func(t *testing.T) {
			path := jlmOf(t, models()[name])
			m, err := model.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			var all uint64
			for i := range m.Container.NPages() {
				all += m.Container.PageBytes(i)
			}
			half := m.Cfg.NLayer / 2
			if half < 1 {
				t.Fatalf("%d blocks: no seam to put on a card", m.Cfg.NLayer)
			}
			ids := []int32{1, 7, 42, 3, 99, 5, 17, 2, 11, 13}
			more := []int32{23, 29, 31, 37}
			g := swaTier(t)
			defer g.Close()
			seam := func(in *Instance) {
				t.Helper()
				in.SetDeviceLayers(g, half)
				if in.GPULayers() != half {
					t.Fatalf("placed %d of %d blocks on the card: %q", in.GPULayers(), half, g.Err())
				}
			}

			alone := testInstance(t, m, len(ids)+1, WithPageBudget(all/2))
			wantCPU := teacherForced(t, alone, ids)
			alone.Close()
			alone = testInstance(t, m, len(ids)+len(more)+1, WithPageBudget(all/2))
			seam(alone)
			wantCard := teacherForced(t, alone, append(slices.Clone(ids), more...))
			alone.Close()

			cpu := sharedCPU(t)
			x := testInstance(t, m, len(ids)+1, cpu, WithPageBudget(all/2))
			y := testInstance(t, m, len(ids)+len(more)+1, cpu, WithPageBudget(all/2))
			defer y.Close()
			seam(y)
			if x.fr != y.fr {
				x.Close()
				t.Fatal("two instances on one CPU device page into two frames: this gate proved nothing")
			}
			if b := x.fr.Budget(); b == 0 || b >= all {
				x.Close()
				t.Fatalf("the shared frames' budget is %d of %d page bytes: nothing pages", b, all)
			}
			var got [2][][]float32
			var wg sync.WaitGroup
			for i, in := range []*Instance{x, y} {
				wg.Add(1)
				go func() { defer wg.Done(); got[i] = teacherForced(t, in, ids) }()
			}
			wg.Wait()
			x.Close()
			got[1] = append(got[1], teacherForced(t, y, more)...)
			if at := sameRows(got[0], wantCPU); at >= 0 {
				t.Errorf("the CPU's instance beside the card's parts from it alone at position %d", at)
			}
			if at := sameRows(got[1], wantCard); at >= 0 {
				t.Errorf("the card's instance beside the CPU's, then after it closed, parts from it alone at position %d", at)
			}
			if fr := y.FramesOn(nn.CPU); fr != y.fr || fr.Budget() == 0 {
				t.Error("the CPU's instance closing took the frames the card's instance pages its CPU blocks into")
			}
		})
	}
}

// TestAQuotaShrinksUnderARunningDecode is the quota-shrinking transition
// (docs/design/instance-paging.md step 6): two instances on one CPU device
// decode at once while their quotas fall to a page and rise back again and
// again; each decodes as an instance alone does, bit for bit -- residency is
// a memory decision, never an arithmetic one -- and once they are out the
// frames hold no more than the quota last set. A budget that evicted a page
// a reader was still in would hand its frame to another page, or back to the
// system, under the reader.
func TestAQuotaShrinksUnderARunningDecode(t *testing.T) {
	for _, name := range transitionModels() {
		t.Run(name, func(t *testing.T) {
			path := jlmOf(t, models()[name])
			m, err := model.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			page := m.PageSize()
			ids := make([]int32, 48)
			for i := range ids {
				ids[i] = int32((7 + 13*i) % m.Cfg.NVocab)
			}
			solo := testInstance(t, m, len(ids)+1)
			want := teacherForced(t, solo, ids)
			solo.Close()

			cpu := sharedCPU(t)
			x := testInstance(t, m, len(ids)+1, cpu)
			defer x.Close()
			y := testInstance(t, m, len(ids)+1, cpu)
			defer y.Close()
			var got [2][][]float32
			var wg sync.WaitGroup
			done := make(chan struct{})
			for i, in := range []*Instance{x, y} {
				wg.Add(1)
				go func() { defer wg.Done(); got[i] = teacherForced(t, in, ids) }()
			}
			go func() { wg.Wait(); close(done) }()
			flips := 0
			for running := true; running; {
				select {
				case <-done:
					running = false
				default:
					b := page
					if flips%2 == 1 {
						b = 0
					}
					x.SetPageBudget(b)
					y.SetPageBudget(b)
					flips++
				}
			}
			if flips < 4 {
				t.Fatalf("the quotas changed %d times while the instances ran: this gate proved nothing", flips)
			}
			for i := range got {
				if at := sameRows(got[i], want); at >= 0 {
					t.Errorf("instance %d parts from one alone at position %d with its quota changing under it", i, at)
				}
			}
			x.SetPageBudget(page)
			y.SetPageBudget(page)
			if r, b := x.fr.ResidentBytes(), x.fr.Budget(); b != page || (r > b && x.fr.ResidentPages() > 1) {
				t.Errorf("out of every reader, the frames hold %d bytes against a budget of %d (want %d)", r, b, page)
			}
			t.Logf("%d quota changes under two decodes of %d tokens", flips, len(ids))
		})
	}
}

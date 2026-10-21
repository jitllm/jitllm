//go:build jitllmtest

package board

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// hostDecode is every kernel main's host JIT called for one decode step after
// a short prompt, each once.
func hostDecode(t *testing.T, name string) []cpu.Called {
	t.Helper()
	m, err := model.Open(testmodels.Path(name+".jlm"), model.WithJITOptions(nn.WithTune(nn.TuneOff)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	s := m.NewState(32)
	defer s.Close()
	lg, err := s.Prefill(m.Vocab.Encode(prompt, true))
	if err != nil {
		t.Fatal(err)
	}
	rec := cpu.Record()
	_, err = s.Forward(model.Greedy(lg))
	called := rec.Stop()
	if err != nil {
		t.Fatal(err)
	}
	if len(called) == 0 {
		t.Fatalf("%s: the host called nothing for the step", name)
	}
	return called
}

func TestHostDecodeKernels(t *testing.T) {
	t.Logf("host tier %s", cpu.HostTier())
	for _, c := range hostDecode(t, *boardModel) {
		t.Logf("%-28s %6d bytes %4d calls", c.Name, len(c.Code), c.Calls)
	}
}

// emitterOf names the emitter of the host's table whose code is c, among
// those that take no shape and the activation kinds.
func emitterOf(em cpu.Emitters, c []byte) string {
	v := reflect.ValueOf(em)
	for i := 0; i < v.NumField(); i++ {
		f := v.Field(i)
		if f.Kind() != reflect.Func || f.IsNil() {
			continue
		}
		ft := f.Type()
		var outs []reflect.Value
		switch {
		case ft.NumIn() == 0:
			outs = f.Call(nil)
		case ft.NumIn() == 1 && ft.In(0) == reflect.TypeOf(cpu.ActKind(0)):
			for k := cpu.ActKind(0); k < 32; k++ {
				o := f.Call([]reflect.Value{reflect.ValueOf(k)})
				if o[1].IsNil() && bytes.Equal(o[0].Bytes(), c) {
					return fmt.Sprintf("%s(%d)", v.Type().Field(i).Name, k)
				}
			}
			continue
		default:
			continue
		}
		if len(outs) == 2 && outs[1].IsNil() && bytes.Equal(outs[0].Bytes(), c) {
			return v.Type().Field(i).Name
		}
	}
	type cand struct {
		name string
		code func() ([]byte, error)
	}
	var cs []cand
	for n := 1; n <= 512; n++ {
		cs = append(cs, cand{fmt.Sprintf("RMSNorm(%d)", n), func() ([]byte, error) { return em.RMSNorm(n) }})
	}
	for p := 1; p <= 128; p++ {
		cs = append(cs, cand{fmt.Sprintf("RopeTable(%d)", p), func() ([]byte, error) { return em.RopeTable(p) }})
	}
	for hd := 8; hd <= 256; hd *= 2 {
		for nrot := 2; nrot <= hd; nrot += 2 {
			for _, neox := range []bool{false, true} {
				cs = append(cs, cand{fmt.Sprintf("RoPE(%d,%d,%v)", hd, nrot, neox), func() ([]byte, error) { return em.RoPE(hd, nrot, neox) }})
			}
		}
		for st := hd; st <= 8*hd; st += hd {
			for kv := cpu.KVF32; kv <= cpu.KVQ8; kv++ {
				for name, f := range map[string]func(int, int, cpu.KVFmt) ([]byte, error){
					"AttnScores": em.AttnScores, "AttnAcc": em.AttnAcc, "AttnAccInto": em.AttnAccInto,
					"AttnScores2": em.AttnScores2, "AttnAcc2": em.AttnAcc2, "AttnAcc2Into": em.AttnAcc2Into} {
					cs = append(cs, cand{fmt.Sprintf("%s(%d,%d,%d)", name, hd, st, kv), func() ([]byte, error) { return f(hd, st, kv) }})
				}
			}
		}
		cs = append(cs, cand{fmt.Sprintf("KVWiden(%d)", hd), func() ([]byte, error) { return em.KVWiden(hd) }})
	}
	for _, b := range []bool{false, true} {
		cs = append(cs, cand{fmt.Sprintf("Widen(%v)", b), func() ([]byte, error) { return em.Widen(b) }},
			cand{fmt.Sprintf("QuantAct(%v)", b), func() ([]byte, error) { return em.QuantAct(b) }})
	}
	for _, x := range cs {
		if b, err := x.code(); err == nil && bytes.Equal(b, c) {
			return x.name
		}
	}
	return "?"
}

var hostDump = flag.String("host-dump", "", "TestHostDecodeEmitters writes each kernel's code into this directory")

func TestHostDecodeEmitters(t *testing.T) {
	em := cpu.EmittersFor(cpu.HostTier())
	for _, c := range hostDecode(t, *boardModel) {
		name := emitterOf(*em, c.Code)
		t.Logf("%-14s %-20s %6d bytes %4d calls", c.Name, name, len(c.Code), c.Calls)
		if *hostDump != "" {
			if err := os.WriteFile(filepath.Join(*hostDump, strings.NewReplacer("(", "_", ")", "", ",", "_").Replace(name)+".bin"), c.Code, 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
}

var hostDiff = flag.String("host-diff", "", "TestBoardHost writes out main's kernels with this name prefix beside the composed kernel nearest each")

// TestBoardHost holds the kernels the composer forms for the host to the
// code main's host JIT ran for the same decode step: each main kernel
// identical where one composed kernel is its bytes, and its instructions
// matched as the longest run in order one composed kernel shares with it.
func TestBoardHost(t *testing.T) {
	em := cpu.EmittersFor(cpu.HostTier())
	plan, err := composed(t, *boardModel, func(jit.Part) jit.Device { return host })
	if err != nil {
		t.Logf("not formed: %v", err)
	}
	var mine [][]byte
	var passes [][]string
	for _, pp := range plan.Parts {
		for _, c := range pp.Codes {
			if c.CPU != nil {
				mine = append(mine, c.CPU)
				passes = append(passes, c.Passes)
			}
		}
	}
	type row struct {
		name           string
		total, matched int
		same           bool
		passes         []string
	}
	var rows []row
	same, total, matched := 0, 0, 0
	for _, c := range hostDecode(t, *boardModel) {
		r := row{name: emitterOf(*em, c.Code)}
		want := disassemble(t, c.Code)
		r.total = len(want)
		var near []string
		for x, m := range mine {
			if bytes.Equal(m, c.Code) {
				r.same = true
			}
			got := disassemble(t, m)
			if n := lcs(want, got); n > r.matched || near == nil {
				r.matched, near, r.passes = max(r.matched, n), got, passes[x]
			}
		}
		if *hostDiff != "" && strings.HasPrefix(r.name, *hostDiff) {
			for i := 0; i < max(len(want), len(near)); i++ {
				var w, g string
				if i < len(want) {
					w = want[i]
				}
				if i < len(near) {
					g = near[i]
				}
				mark := " "
				if w != g {
					mark = "*"
				}
				t.Logf("%s %-44s %s", mark, w, g)
			}
		}
		if r.same {
			same++
		}
		total += r.total
		matched += r.matched
		rows = append(rows, r)
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].total-rows[i].matched > rows[j].total-rows[j].matched })
	for _, r := range rows {
		t.Logf("%-22s %4d of %4d instructions identical=%v passes %v", r.name, r.matched, r.total, r.same, r.passes)
	}
	t.Logf("%s on the host (%s): %d of main's %d instructions (%.1f%%), %d of main's %d kernels identical; %d composed kernels",
		*boardModel, cpu.HostTier(), matched, total, 100*float64(matched)/float64(max(total, 1)), same, len(rows), len(mine))
}

// disassemble is code's instructions, each its mnemonic and operands.
func disassemble(t *testing.T, code []byte) []string {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "k*.bin")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(code); err != nil {
		t.Fatal(err)
	}
	f.Close()
	out, err := exec.Command("objdump", "-D", "-b", "binary", "-mi386:x86-64", "--no-show-raw-insn", f.Name()).Output()
	if err != nil {
		t.Fatalf("the board reads code through objdump: %v", err)
	}
	var ins []string
	for _, l := range strings.Split(string(out), "\n") {
		if _, rest, ok := strings.Cut(l, ":\t"); ok && strings.HasPrefix(strings.TrimSpace(l), strings.TrimSpace(strings.SplitN(l, ":", 2)[0])) {
			ins = append(ins, strings.Join(strings.Fields(rest), " "))
		}
	}
	return ins
}

// lcs is the length of the longest sequence a and b share in order.
func lcs(a, b []string) int {
	prev, cur := make([]int, len(b)+1), make([]int, len(b)+1)
	for i := range a {
		for j := range b {
			if a[i] == b[j] {
				cur[j+1] = prev[j] + 1
			} else {
				cur[j+1] = max(cur[j], prev[j+1])
			}
		}
		prev, cur = cur, prev
	}
	return prev[len(b)]
}

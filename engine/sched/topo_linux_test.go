//go:build linux

package sched

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"unsafe"
)

// TestTopologyClassifiesTheMachinesThisEngineRunsOn reads synthetic sysfs
// trees for the four machine shapes this engine has, plus the hybrid corner,
// and checks PCores, ECores and SMTSiblings on each.
//
// The Atom row matters: a Goldmont Atom has four single-threaded cores and no
// SMT, and "single-threaded => E-core" would leave PCores() empty.
func TestTopologyClassifiesTheMachinesThisEngineRunsOn(t *testing.T) {
	ok := host().allowed()
	keep := func(cpus []int) []int {
		var out []int
		for _, c := range cpus {
			if len(ok) == 0 || ok[c] {
				out = append(out, c)
			}
		}
		return out
	}
	// A P-core is taken at the first of its threads the mask allows (see
	// TestPCoresTakesTheThreadTheMaskAllows), so under a taskset that drops a
	// core's first thread its expected CPU is the sibling the mask keeps, not
	// nothing: keepCores maps each listed P-core to that thread.
	keepCores := func(sibs []string, p []int) []int {
		isP := map[string]bool{}
		for _, c := range p {
			isP[sibs[c]] = true
		}
		var out []int
		seen := map[string]bool{}
		for cpu, s := range sibs {
			if len(ok) > 0 && !ok[cpu] || seen[s] {
				continue
			}
			seen[s] = true
			if isP[s] {
				out = append(out, cpu)
			}
		}
		return out
	}
	pairs := func(n int) []string { // cpu 2i and 2i+1 are one core, Alder Lake style
		var s []string
		for i := 0; i < n; i++ {
			s = append(s, fmt.Sprintf("%d-%d", i&^1, i|1))
		}
		return s
	}
	singles := func(from, n int) []string {
		var s []string
		for i := from; i < from+n; i++ {
			s = append(s, fmt.Sprint(i))
		}
		return s
	}
	cases := []struct {
		name      string
		sibs      []string
		atoms     string // "" = the kernel publishes no hybrid split
		p, e, smt []int
	}{
		{"alder lake, 6P+8E, kernel publishes cpu_atom",
			append(pairs(12), singles(12, 8)...), "12-19",
			[]int{0, 2, 4, 6, 8, 10}, []int{12, 13, 14, 15, 16, 17, 18, 19},
			[]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}},
		{"alder lake, older kernel without cpu_atom",
			append(pairs(12), singles(12, 8)...), "",
			[]int{0, 2, 4, 6, 8, 10}, []int{12, 13, 14, 15, 16, 17, 18, 19},
			[]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}},
		{"goldmont atom C3558, 4 cores, no SMT",
			singles(0, 4), "",
			[]int{0, 1, 2, 3}, nil, []int{0, 1, 2, 3}},
		{"xeon D-2123IT, 4 cores x 2 threads, siblings 0/4",
			[]string{"0,4", "1,5", "2,6", "3,7", "0,4", "1,5", "2,6", "3,7"}, "",
			[]int{0, 1, 2, 3}, nil, []int{0, 4, 1, 5, 2, 6, 3, 7}},
		{"hybrid with SMT disabled, kernel publishes cpu_atom",
			singles(0, 10), "6-9",
			[]int{0, 1, 2, 3, 4, 5}, []int{6, 7, 8, 9}, []int{0, 1, 2, 3, 4, 5}},
	}
	m := host()
	for _, c := range cases {
		root := t.TempDir()
		for cpu, s := range c.sibs {
			d := filepath.Join(root, "devices/system/cpu", fmt.Sprintf("cpu%d", cpu), "topology")
			if err := os.MkdirAll(d, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(d, "thread_siblings_list"), []byte(s+"\n"), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		if c.atoms != "" {
			d := filepath.Join(root, "devices/cpu_atom")
			if err := os.MkdirAll(d, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(d, "cpus"), []byte(c.atoms+"\n"), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		m.sys = root
		if got, want := m.pCores(), keepCores(c.sibs, c.p); !reflect.DeepEqual(got, want) {
			t.Errorf("%s: PCores = %v, want %v", c.name, got, want)
		}
		if got, want := m.eCores(), keep(c.e); !reflect.DeepEqual(got, want) {
			t.Errorf("%s: ECores = %v, want %v", c.name, got, want)
		}
		if got, want := m.smtSiblings(), keep(c.smt); !reflect.DeepEqual(got, want) {
			t.Errorf("%s: SMTSiblings = %v, want %v", c.name, got, want)
		}
		if p, want := DecodeCores(onMachine(m)), keepCores(c.sibs, c.p); len(want) > 0 && len(p) != len(want) {
			t.Errorf("%s: DecodeCores = %v, want every P-core %v", c.name, p, want)
		}
	}
}

// TestPCoresTakesTheThreadTheMaskAllows: a process confined to the second
// thread of each core still gets one worker per core. PCores once marked a core
// seen at its first thread before asking whether the mask allows it, so a
// taskset of only the hyperthreads found no P-core at all and DecodeCores fell
// back to a one-worker pool outside the mask.
func TestPCoresTakesTheThreadTheMaskAllows(t *testing.T) {
	if runtime.NumCPU() < 8 {
		t.Skipf("%d CPUs: the synthetic xeon's threads 4-7 are not on this machine -- this gate proved nothing", runtime.NumCPU())
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	setMask := func(cpus map[int]bool) {
		t.Helper()
		var mask [16]uint64
		for c := range cpus {
			mask[c/64] |= 1 << uint(c%64)
		}
		if _, _, errno := syscall.RawSyscall(syscall.SYS_SCHED_SETAFFINITY,
			0, uintptr(len(mask)*8), uintptr(unsafe.Pointer(&mask[0]))); errno != 0 {
			t.Fatalf("sched_setaffinity: %v", errno)
		}
	}
	was := host().allowed()
	if !was[4] || !was[5] || !was[6] || !was[7] {
		t.Skipf("the mask %v does not hold CPUs 4-7 -- this gate proved nothing", was)
	}
	defer setMask(was)
	root := t.TempDir()
	for cpu, s := range []string{"0,4", "1,5", "2,6", "3,7", "0,4", "1,5", "2,6", "3,7"} {
		d := filepath.Join(root, "devices/system/cpu", fmt.Sprintf("cpu%d", cpu), "topology")
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(d, "thread_siblings_list"), []byte(s+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	m := host()
	m.sys = root
	setMask(map[int]bool{4: true, 5: true, 6: true, 7: true})
	if got, want := m.pCores(), []int{4, 5, 6, 7}; !reflect.DeepEqual(got, want) {
		t.Fatalf("confined to the second thread of each core, PCores = %v, want %v", got, want)
	}
	if got := DecodeCores(onMachine(m)); len(got) != 4 {
		t.Fatalf("DecodeCores = %v: four cores are in the mask", got)
	}
}

// TestPoolWidthWithoutAnAffinityMask stands in for a sandbox (gVisor)
// that answers neither sched_getaffinity nor /sys topology. PCores once returned
// nothing there and DecodeCores fell back to one CPU, so the shipped engine
// decoded on one worker. The pool now takes the online list, cut to the cgroup
// quota, and with a mask and no topology the mask.
func TestPoolWidthWithoutAnAffinityMask(t *testing.T) {
	tree := func(online, cpuMax string, sibs []string) string {
		root := t.TempDir()
		write := func(rel, s string) {
			p := filepath.Join(root, rel)
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(p, []byte(s+"\n"), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		if online != "" {
			write("devices/system/cpu/online", online)
		}
		if cpuMax != "" {
			write("fs/cgroup/cpu.max", cpuMax)
		}
		for cpu, s := range sibs {
			write(fmt.Sprintf("devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu), s)
		}
		return root
	}
	for _, c := range []struct {
		name   string
		root   string
		mask   map[int]bool
		want   int
		source string
	}{
		{"online list, no quota", tree("0-7", "", nil), nil, 8, "no affinity mask"},
		{"online list, quota of 3.5", tree("0-7", "350000 100000", nil), nil, 4, "cgroup quota 4"},
		{"unlimited quota", tree("0-5", "max 100000", nil), nil, 6, "no affinity mask"},
		{"topology, quota of 2", tree("", "200000 100000", []string{"0,4", "1,5", "2,6", "3,7", "0,4", "1,5", "2,6", "3,7"}), nil, 2, "sysfs topology"},
		{"no topology, a mask", tree("0-7", "", nil), map[int]bool{1: true, 3: true, 5: true}, 3, "affinity mask"},
	} {
		t.Run(c.name, func(t *testing.T) {
			m := host()
			m.sys, m.affinity = c.root, func() map[int]bool { return c.mask }
			d := DecodeCores(onMachine(m))
			if len(d) != c.want {
				t.Fatalf("DecodeCores = %v, want %d CPUs (source: %s)", d, c.want, m.coreSource())
			}
			p := New(d)
			defer p.Close()
			if p.N() != c.want {
				t.Fatalf("pool of %d participants, want %d", p.N(), c.want)
			}
			if src := m.coreSource(); !strings.Contains(src, c.source) {
				t.Fatalf("CoreSource = %q, want it to name %q", src, c.source)
			}
			t.Logf("%d workers from %s", p.N(), m.coreSource())
		})
	}
}

// onMachine reads the pool's cores from m rather than the running machine.
func onMachine(m machine) Option { return func(c *config) { c.m = m } }

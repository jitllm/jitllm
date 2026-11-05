//go:build amd64 || arm64

package engine

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Helpers shared by the measurement instruments (behind the jitllmbench tag) and
// by the deterministic gates beside them (labelcost_test.go), so they live on
// the side that is always built.

func envInt(name string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(name)); err == nil && v >= 0 {
		return v
	}
	return def
}

func envOr(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

// coreBusy is the busy percentage of the CPUs this process is pinned to,
// sampled from /proc/stat. It returns an error where /proc/stat is unavailable
// (darwin), which is the caller's cue to carry on rather than to refuse.
//
// PSI is blind to the tenant that ruins a row: another project's `go test` on
// the pinned cores can read 87.7% busy while PSI reads 2.35%. The question is
// whether the cores about to be pinned are busy, which /proc/stat answers
// directly. scripts/corebusy.py is the same measurement for the shell
// harnesses.
func coreBusy(d time.Duration) (float64, error) {
	// The affinity comes from /proc/self/status rather than a syscall so this
	// needs no dependency the rest of the tree does not already have.
	st, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0, err
	}
	mask := map[int]bool{}
	for _, ln := range strings.Split(string(st), "\n") {
		v, ok := strings.CutPrefix(ln, "Cpus_allowed_list:")
		if !ok {
			continue
		}
		for _, part := range strings.Split(strings.TrimSpace(v), ",") {
			lo, hi, isRange := strings.Cut(part, "-")
			a, err := strconv.Atoi(lo)
			if err != nil {
				continue
			}
			b := a
			if isRange {
				if b, err = strconv.Atoi(hi); err != nil {
					continue
				}
			}
			for c := a; c <= b; c++ {
				mask[c] = true
			}
		}
	}
	if len(mask) == 0 {
		return 0, fmt.Errorf("no Cpus_allowed_list")
	}
	snap := func() (busy, total uint64, err error) {
		b, err := os.ReadFile("/proc/stat")
		if err != nil {
			return 0, 0, err
		}
		for _, ln := range strings.Split(string(b), "\n") {
			if !strings.HasPrefix(ln, "cpu") || ln == "" || ln[3] == ' ' {
				continue
			}
			f := strings.Fields(ln)
			n, err := strconv.Atoi(strings.TrimPrefix(f[0], "cpu"))
			if err != nil || !mask[n] {
				continue
			}
			var sum, idle uint64
			for i, v := range f[1:] {
				u, _ := strconv.ParseUint(v, 10, 64)
				sum += u
				if i == 3 || i == 4 { // idle, iowait
					idle += u
				}
			}
			busy += sum - idle
			total += sum
		}
		return busy, total, nil
	}
	b0, t0, err := snap()
	if err != nil {
		return 0, err
	}
	time.Sleep(d)
	b1, t1, err := snap()
	if err != nil {
		return 0, err
	}
	if t1 == t0 {
		return 0, nil
	}
	return 100 * float64(b1-b0) / float64(t1-t0), nil
}

func median(v []float64) float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	n := len(s)
	if n == 0 {
		return 0
	}
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

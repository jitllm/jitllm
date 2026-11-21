//go:build linux

package probe

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// cacheSizes is the data caches, nearest first, as Linux states them for
// the first CPU: a level is shared where more than one CPU's threads that
// are not the first's siblings share it -- a list longer than two CPUs.
func cacheSizes() ([]cache, error) {
	var out []cache
	for i := 0; ; i++ {
		dir := fmt.Sprintf("/sys/devices/system/cpu/cpu0/cache/index%d/", i)
		typ, err := os.ReadFile(dir + "type")
		if err != nil {
			break
		}
		if strings.TrimSpace(string(typ)) == "Instruction" {
			continue
		}
		b, err := os.ReadFile(dir + "size")
		if err != nil {
			return nil, err
		}
		n, err := parseSize(strings.TrimSpace(string(b)))
		if err != nil {
			return nil, err
		}
		list, err := os.ReadFile(dir + "shared_cpu_list")
		if err != nil {
			return nil, err
		}
		out = append(out, cache{bytes: n, shared: cpuCount(strings.TrimSpace(string(list))) > 2})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("probe: no cache sizes under /sys/devices/system/cpu/cpu0/cache")
	}
	return out, nil
}

// cpuCount is how many CPUs a list like "0-1,4" names.
func cpuCount(list string) int {
	n := 0
	for _, part := range strings.Split(list, ",") {
		lo, hi, ok := strings.Cut(part, "-")
		if !ok {
			n++
			continue
		}
		a, errA := strconv.Atoi(lo)
		b, errB := strconv.Atoi(hi)
		if errA == nil && errB == nil {
			n += b - a + 1
		}
	}
	return n
}

func parseSize(s string) (int64, error) {
	mul := int64(1)
	switch {
	case strings.HasSuffix(s, "K"):
		mul, s = 1<<10, strings.TrimSuffix(s, "K")
	case strings.HasSuffix(s, "M"):
		mul, s = 1<<20, strings.TrimSuffix(s, "M")
	}
	n, err := strconv.ParseInt(s, 10, 64)
	return n * mul, err
}

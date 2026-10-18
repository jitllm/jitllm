package server

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

// hostTotal and hostAvailable read /proc/meminfo directly rather than adding a
// dependency. On a host with no /proc they return 0, which callers treat as
// "not reported"; nothing divides by these.
func hostTotal() uint64 { return meminfoField("MemTotal:") }

// hostAvailable is the quantity that moves; hostTotal is read each time too,
// since neither is on a hot path.
func hostAvailable() uint64 { return meminfoField("MemAvailable:") }

func meminfoField(key string) uint64 {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, key) {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0
		}
		kb, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			return 0
		}
		return kb * 1024
	}
	return 0
}

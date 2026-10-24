package x86

import "strings"

type ISA uint16

const (
	ISASSE2 ISA = 1 << iota
	ISASSE3
	ISASSSE3
	ISASSE41
	ISASSE42
	ISAPOPCNT
	ISAAVX2
	ISAFMA
	ISAF16C
	ISAVNNI
	ISABMI2
)

const (
	ISATierSSE  = ISASSE2 | ISASSE3 | ISASSSE3 | ISASSE41
	ISATierAVX2 = ISAAVX2 | ISAFMA | ISAF16C
)

var isaNames = [...]string{"sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx2", "fma", "f16c", "avx-vnni", "bmi2"}

func (s ISA) Names() []string {
	var out []string
	for i, n := range isaNames {
		if s&(1<<i) != 0 {
			out = append(out, n)
		}
	}
	return out
}

func (s ISA) String() string {
	if s == 0 {
		return "none"
	}
	return strings.Join(s.Names(), ", ")
}

type regUse struct {
	vecHigh int
	stack   int
	vex     bool
	touched bool
}

func (r *regUse) vec(rs ...Reg) {
	if !r.touched {
		r.vecHigh, r.touched = -1, true
	}
	for _, x := range rs {
		r.vecHigh = max(r.vecHigh, int(x))
	}
}

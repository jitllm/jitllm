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

const isaCount = 11

func isaName(i int) string {
	switch i {
	case 0:
		return "sse2"
	case 1:
		return "sse3"
	case 2:
		return "ssse3"
	case 3:
		return "sse4.1"
	case 4:
		return "sse4.2"
	case 5:
		return "popcnt"
	case 6:
		return "avx2"
	case 7:
		return "fma"
	case 8:
		return "f16c"
	case 9:
		return "avx-vnni"
	}
	return "bmi2"
}

func (s ISA) Names() []string {
	var out []string
	for i := 0; i < isaCount; i++ {
		if s&(1<<i) != 0 {
			out = append(out, isaName(i))
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

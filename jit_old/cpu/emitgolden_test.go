package cpu

import (
	"bufio"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// The emitter golden gate: the machine code a family of emitters writes,
// recorded per case from a tree before its emitters were restructured. A
// refactor of the code that writes kernels must leave every case identical;
// a case that differs names the first differing offset. Goldens are written
// only by -update-emit-golden, from a tree whose emitters are known good --
// blessing a golden made from a changed emitter proves nothing (RULE 11b).

var updateEmitGolden = flag.Bool("update-emit-golden", false, "rewrite the emitter goldens from the current emitters")

type goldenCase struct {
	name string
	code []byte
}

// normWidths is every width the norm family is gated at: 1..65 (every tail
// length against zero to two unrolled bodies), the boundaries around the
// 16- and 32-element bodies, and the widths real models use.
func normWidths() []int {
	var w []int
	for n := 1; n <= 65; n++ {
		w = append(w, n)
	}
	w = append(w, 71, 72, 73, 95, 96, 97, 100, 127, 128, 129, 160, 191, 192, 193,
		224, 255, 256, 257, 288, 320, 384, 511, 512, 513, 576, 640, 768, 1000,
		1024, 1152, 1280, 1536, 1600, 2048, 2304, 2560, 3072, 3584, 4096, 4097,
		5120, 6144, 7168, 8192)
	return w
}

func goldenPath(family string) string {
	return filepath.Join("testdata", "emit_golden_"+family+".txt.gz")
}

func readGolden(t *testing.T, family string) map[string][]byte {
	t.Helper()
	f, err := os.Open(goldenPath(family))
	if err != nil {
		t.Fatalf("golden %s: %v (capture it with -update-emit-golden on known-good emitters)", family, err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	defer zr.Close()
	out := map[string][]byte{}
	sc := bufio.NewScanner(zr)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		p := strings.Fields(sc.Text())
		if len(p) != 4 {
			t.Fatalf("golden %s: malformed line %q", family, sc.Text())
		}
		b, err := hex.DecodeString(p[3])
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(b)
		if hex.EncodeToString(sum[:]) != p[2] || fmt.Sprint(len(b)) != p[1] {
			t.Fatalf("golden %s: case %s is corrupt", family, p[0])
		}
		out[p[0]] = b
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return out
}

func writeGolden(t *testing.T, family string, cases []goldenCase) {
	t.Helper()
	if err := os.MkdirAll("testdata", 0o755); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(goldenPath(family))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zw, err := gzip.NewWriterLevel(f, gzip.BestCompression)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		sum := sha256.Sum256(c.code)
		fmt.Fprintf(zw, "%s %d %s %s\n", c.name, len(c.code), hex.EncodeToString(sum[:]), hex.EncodeToString(c.code))
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
}

// checkGolden compares cases against the recorded family. Every recorded case
// must be produced and every produced case must be recorded, so a gate that
// quietly stopped emitting something fails rather than shrinking.
func checkGolden(t *testing.T, family string, cases []goldenCase) {
	t.Helper()
	if len(cases) == 0 {
		t.Fatal("no cases emitted")
	}
	if *updateEmitGolden {
		writeGolden(t, family, cases)
		return
	}
	want := readGolden(t, family)
	seen := map[string]bool{}
	bad := 0
	for _, c := range cases {
		seen[c.name] = true
		w, ok := want[c.name]
		if !ok {
			t.Errorf("%s: no golden", c.name)
			bad++
			continue
		}
		if string(w) == string(c.code) {
			continue
		}
		bad++
		off := 0
		for off < len(w) && off < len(c.code) && w[off] == c.code[off] {
			off++
		}
		t.Errorf("%s: emitted code differs at offset %d (golden %d bytes, now %d bytes)", c.name, off, len(w), len(c.code))
		if bad >= 20 {
			t.Fatalf("%s: stopping after 20 differing cases", family)
		}
	}
	var missing []string
	for name := range want {
		if !seen[name] {
			missing = append(missing, name)
		}
	}
	sort.Strings(missing)
	if len(missing) > 0 {
		t.Errorf("%s: golden cases no longer emitted, first %s (%d in all)", family, missing[0], len(missing))
	}
}

// normGoldenCases emits the row-reduction family through one ISA's entry
// points. rms and stat return the kernel for width n; the case names are the
// same on every ISA.
func normGoldenCases(t *testing.T, rms func(n int) ([]byte, error), stat func(n int, bias bool, mode RowMode) ([]byte, error)) []goldenCase {
	t.Helper()
	var cases []goldenCase
	add := func(name string, code []byte, err error) {
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if len(code) == 0 {
			t.Fatalf("%s: emitted nothing", name)
		}
		cases = append(cases, goldenCase{name, code})
	}
	for _, n := range normWidths() {
		c, err := rms(n)
		add(fmt.Sprintf("rmsnorm/n=%d", n), c, err)
		for _, bias := range []bool{false, true} {
			c, err = stat(n, bias, RowLayerNorm)
			add(fmt.Sprintf("layernorm/bias=%v/n=%d", bias, n), c, err)
		}
		c, err = stat(n, false, RowGauss)
		add(fmt.Sprintf("gausstopk/n=%d", n), c, err)
		c, err = stat(n, false, RowMag)
		add(fmt.Sprintf("magmatch/n=%d", n), c, err)
	}
	return cases
}

// TestNEONNormEmittersAreByteStable runs on every host: the NEON assembler
// builds anywhere, so amd64 emits the arm64 bytes.
func TestNEONNormEmittersAreByteStable(t *testing.T) {
	checkGolden(t, "neon_norm", normGoldenCases(t,
		func(n int) ([]byte, error) { return EmitA64RMSNorm(n), nil },
		func(n int, bias bool, mode RowMode) ([]byte, error) { return emitA64RowStat(n, bias, mode), nil }))
}

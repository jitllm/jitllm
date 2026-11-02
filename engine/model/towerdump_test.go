package model

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// The tower's node-by-node oracle: llama-mtmd-debug's graph dump.
//
// `llama-mtmd-debug -p encode --image rainbow -n <ImageSz>` builds a picture in
// memory as raw floats and runs clip's graph with an eval callback on every
// node, printing the first and last three values along each axis and the sum
// of the whole tensor. Raw floats means no decode, resize or normalisation is
// in the comparison: only the tower's arithmetic. scripts/vlmgold.py writes
// the dumps these tests read ($JITLLM_MODELS/vlm/oracle/<family>-rainbow.txt).

// mtmdNode is one 2-D node of the dump: rows are ggml's ne1 (positions), cols
// its ne0 (channels).
type mtmdNode struct {
	width, rows  int
	rowIdx, cols []int
	vals         [][]float64 // [len(rowIdx)][len(cols)]
	sum          float64
}

var mtmdHead = regexp.MustCompile(`common_debug_cb_eval:\s+(.+?) = \(f32\).*= \{(\d+), (\d+), 1, 1\}$`)

// parseMtmdDump reads every 2-D f32 node of a dump, keyed by its name. A name
// printed twice keeps its first node.
func parseMtmdDump(t testing.TB, path string) map[string]*mtmdNode {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		testmodels.Missing(t, "ORACLE MISSING: %v -- RULE 11: run scripts/towergold.sh to produce it; "+
			"without it this gate proves nothing", err)
	}
	defer f.Close()
	out := map[string]*mtmdNode{}
	var cur *mtmdNode
	var name string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if m := mtmdHead.FindStringSubmatch(line); m != nil {
			w, _ := strconv.Atoi(m[2])
			r, _ := strconv.Atoi(m[3])
			cur, name = &mtmdNode{width: w, rows: r}, strings.TrimSpace(m[1])
			cur.rowIdx, cur.cols = edgeIdx(r), edgeIdx(w)
			continue
		}
		if strings.HasPrefix(line, "common_debug_cb_eval:") {
			cur = nil
			continue
		}
		if cur == nil {
			continue
		}
		tl := strings.TrimSpace(line)
		if v, ok := strings.CutPrefix(tl, "sum = "); ok {
			cur.sum, _ = strconv.ParseFloat(v, 64)
			if _, dup := out[name]; !dup && len(cur.vals) == len(cur.rowIdx) {
				out[name] = cur
			}
			cur = nil
			continue
		}
		if !strings.HasPrefix(tl, "[") || strings.HasPrefix(tl, "[\t") || tl == "[" {
			continue
		}
		var row []float64
		for _, fld := range strings.FieldsFunc(tl, func(r rune) bool { return r == '[' || r == ']' || r == ',' || r == ' ' }) {
			if fld == "..." {
				continue
			}
			x, err := strconv.ParseFloat(fld, 64)
			if err != nil {
				row = nil
				break
			}
			row = append(row, x)
		}
		if len(row) == len(cur.cols) {
			cur.vals = append(cur.vals, row)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return out
}

// edgeIdx is the indices the dump prints along an axis of length n: all of
// them up to six, else the first and last three.
func edgeIdx(n int) []int {
	if n <= 6 {
		out := make([]int, n)
		for i := range out {
			out[i] = i
		}
		return out
	}
	return []int{0, 1, 2, n - 3, n - 2, n - 1}
}

// against holds rows x (row-major, width wide) to a node: the relative RMS over
// the printed values and the relative error of the whole-tensor sum.
func (nd *mtmdNode) against(x []float32) (relRMS, sumRel float64, err error) {
	if len(x) != nd.width*nd.rows {
		return 0, 0, fmt.Errorf("%d floats against a %dx%d node", len(x), nd.rows, nd.width)
	}
	var sse, sy2, sum float64
	for i, r := range nd.rowIdx {
		for j, cl := range nd.cols {
			g, w := float64(x[r*nd.width+cl]), nd.vals[i][j]
			sse += (g - w) * (g - w)
			sy2 += w * w
		}
	}
	for _, v := range x {
		sum += float64(v)
	}
	if sy2 <= 0 || math.IsNaN(sse) || math.IsInf(sse, 0) || math.IsNaN(sum) {
		return 0, 0, fmt.Errorf("degenerate comparison: sse %g sy2 %g sum %g", sse, sy2, sum)
	}
	return math.Sqrt(sse / sy2), math.Abs(sum-nd.sum) / math.Max(math.Abs(nd.sum), 1e-30), nil
}

// rainbow is llama-mtmd-debug's "rainbow" picture, [y][x][channel] raw floats
// in [0, 1], in float32 as it computes it. It is asymmetric in both axes,
// which a checkerboard is not, so a transposed or permuted grid shows.
func rainbow(n int) []float32 {
	px := make([]float32, n*n*3)
	cx, cy := float32(n)/2, float32(n)/2
	maxDist := float32(math.Sqrt(float64(cx*cx + cy*cy)))
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			dx, dy := float32(x)-cx, float32(y)-cy
			hue := float32(math.Atan2(float64(dy), float64(dx))) / (2 * 3.14159265)
			if hue < 0 {
				hue++
			}
			sat := float32(math.Sqrt(float64(dx*dx+dy*dy))) / maxDist
			if sat > 1 {
				sat = 1
			}
			h6 := hue * 6
			i6 := int(h6)
			f := h6 - float32(i6)
			p, q, tt := 1-sat, 1-sat*f, 1-sat*(1-f)
			var r, g, b float32
			switch i6 % 6 {
			case 0:
				r, g, b = 1, tt, p
			case 1:
				r, g, b = q, 1, p
			case 2:
				r, g, b = p, 1, tt
			case 3:
				r, g, b = p, q, 1
			case 4:
				r, g, b = tt, p, 1
			default:
				r, g, b = 1, p, q
			}
			o := (y*n + x) * 3
			px[o], px[o+1], px[o+2] = r, g, b
		}
	}
	return px
}

// towerTrace runs one encode and keeps a copy of every traced point: the
// stages by name and block li's output as "layer_out-li".
func towerTrace(t testing.TB, s *Instance, px []float32) map[string][]float32 {
	t.Helper()
	got := map[string][]float32{}
	s.vis.afterBlock = func(li int, x []float32) {
		got[fmt.Sprintf("layer_out-%d", li)] = append([]float32(nil), x...)
	}
	s.vis.atStage = func(name string, x []float32) { got[name] = append([]float32(nil), x...) }
	defer func() { s.vis.afterBlock, s.vis.atStage = nil, nil }()
	if _, err := s.Encode(px); err != nil {
		t.Fatal(err)
	}
	return got
}

// patchGrid is scripts/visiongold.py's grid: four colours in every 2x2 block of
// patches, dimmed by a diagonal ramp, raw floats [y][x][channel]. On it the
// order inside a pixel shuffle's group is the whole difference, which the
// smooth rainbow forgives.
func patchGrid(n, patch int) []float32 {
	pal := [4][3]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}}
	px := make([]float32, n*n*3)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			k := ((y/patch)%2)*2 + (x/patch)%2
			ramp := float32(0.25) + float32(0.75)*float32(x+y)/float32(2*n)
			for ch := 0; ch < 3; ch++ {
				px[(y*n+x)*3+ch] = pal[k][ch] * ramp
			}
		}
	}
	return px
}

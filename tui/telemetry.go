package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/jitllm/jitllm/common/session"
)

// telemetry is what the engine has reported over time: the rates' recent
// history and when each block last moved. The root feeds it every engine
// message; the sidebar and the engine screen draw from it.
type telemetry struct {
	decode, prompt series
	lastSample     time.Time

	blocks []byte
	// moved is the frame each block last changed device, so a move flashes.
	moved []int
}

const flashFrames = fps

func (t *telemetry) observe(env *env, msg engineMsg) {
	st := env.f.st
	if msg.changed.has(fRates) {
		// The rate means decode once the reply has text: before that it is
		// the first token's arrival over a few microseconds.
		if r := st.DecodeTokS.Get(); r > 0 && st.Streaming.Get() && len(st.Stream.Get()) >= 16 &&
			time.Since(t.lastSample) >= time.Second/2 {
			t.decode.push(r)
			t.lastSample = time.Now()
		}
		if r := st.PromptTokS.Get(); r > 0 && (len(t.prompt.v) == 0 || t.prompt.v[len(t.prompt.v)-1] != r) {
			t.prompt.push(r)
		}
	}
	if msg.changed.has(fBlocks) {
		bm := st.BlockMap.Get()
		if len(bm) != len(t.blocks) {
			t.blocks = append([]byte(nil), bm...)
			t.moved = make([]int, len(bm))
			for i := range t.moved {
				t.moved[i] = -flashFrames
			}
			return
		}
		for i := range bm {
			if session.Placement(bm[i]).Device() != session.Placement(t.blocks[i]).Device() {
				t.moved[i] = env.frame
			}
			t.blocks[i] = bm[i]
		}
	}
}

// flashing is true while a block that moved is still fading.
func (t *telemetry) flashing(frame int) bool {
	for _, f := range t.moved {
		if frame-f < flashFrames {
			return true
		}
	}
	return false
}

// series is a rate's recent history.
type series struct{ v []float64 }

const seriesLen = 240

func (s *series) push(x float64) {
	s.v = append(s.v, x)
	if len(s.v) > seriesLen {
		s.v = s.v[len(s.v)-seriesLen:]
	}
}

func (s *series) stats() (last, peak, mean float64) {
	if len(s.v) == 0 {
		return
	}
	for _, x := range s.v {
		peak = max(peak, x)
		mean += x
	}
	return s.v[len(s.v)-1], peak, mean / float64(len(s.v))
}

func sparks() []rune {
	return []rune("▁▂▃▄▅▆▇█")
}

// chart is the last w samples as bars h rows tall, scaled to the peak.
func (s *series) chart(w, h int, c lipgloss.Color) string {
	if len(s.v) < 6 || h < 1 || w < 1 {
		return ""
	}
	v := s.v[max(0, len(s.v)-w):]
	_, peak, _ := s.stats()
	rows := make([]strings.Builder, h)
	for _, x := range v {
		eighths := 0
		if peak > 0 {
			eighths = int(x/peak*float64(h*8) + 0.5)
		}
		for r := range h {
			lvl := eighths - (h-1-r)*8
			st := lipgloss.NewStyle().Foreground(lerp(cLine, c, 0.3+0.7*float64(h-r)/float64(h)))
			switch {
			case lvl >= 8:
				rows[r].WriteString(st.Render("█"))
			case lvl <= 0:
				rows[r].WriteString(" ")
			default:
				rows[r].WriteString(st.Render(string(sparks()[lvl-1])))
			}
		}
	}
	out := make([]string, h)
	for i := range rows {
		out[i] = rows[i].String()
	}
	return strings.Join(out, "\n")
}

// grid draws one cell per block: a device's colour where a device runs it,
// green on the CPU device when its page is resident, hollow when it is read from
// the file on its next pass; a block that just moved flashes white. The big
// grid sizes its cells to the width, two rows tall and numbered.
func (t *telemetry) grid(bm []byte, w int, big bool, frame int) string {
	cw, rows, gutter := 1, 1, 0
	if big {
		gutter = 5
		cw = min(6, max(2, (w-gutter)/max(1, len(bm))-1))
		rows = 2
	}
	per := max(1, (w-gutter+1)/(cw+1))
	var lines []string
	for start := 0; start < len(bm); start += per {
		end := min(len(bm), start+per)
		for r := range rows {
			var b strings.Builder
			if big {
				label := "     "
				if r == 0 {
					label = fmt.Sprintf("%3d  ", start)
				}
				b.WriteString(sDim().Render(label))
			}
			for i := start; i < end; i++ {
				if i > start {
					b.WriteString(" ")
				}
				b.WriteString(t.cell(bm[i], i, cw, big, frame))
			}
			lines = append(lines, b.String())
		}
		if big && end < len(bm) {
			lines = append(lines, "")
		}
	}
	return strings.Join(lines, "\n")
}

func (t *telemetry) cell(raw byte, i, cw int, big bool, frame int) string {
	p := session.Placement(raw)
	col, glyph := cBrand, "■"
	if big {
		glyph = strings.Repeat("█", cw)
	}
	switch {
	case p.Device() >= 0:
		col = colourOf(p.Device(), deviceColours())
	case p.Resident():
	default:
		col = cLine
		glyph = "□"
		if big {
			glyph = strings.Repeat("░", cw)
		}
	}
	if i < len(t.moved) {
		if age := frame - t.moved[i]; age >= 0 && age < flashFrames {
			col = lerp("#ffffff", col, float64(age)/flashFrames)
		}
	}
	return lipgloss.NewStyle().Foreground(col).Render(glyph)
}

func devBlocks(bm []byte) int {
	n := 0
	for _, b := range bm {
		if session.Placement(b).Device() >= 0 {
			n++
		}
	}
	return n
}

func blockLegend(bm []byte) string {
	var host, paged int
	dev := map[int]int{}
	for _, raw := range bm {
		p := session.Placement(raw)
		switch {
		case p.Device() >= 0:
			dev[p.Device()]++
		case p.Resident():
			host++
		default:
			paged++
		}
	}
	sq := func(c lipgloss.Color) string { return lipgloss.NewStyle().Foreground(c).Render("■") }
	parts := []string{sq(cBrand) + sDim().Render(fmt.Sprintf(" cpu %d", host))}
	for k := range deviceColours() {
		if n := dev[k]; n > 0 {
			parts = append(parts, sq(colourOf(k, deviceColours()))+sDim().Render(fmt.Sprintf(" gpu%d %d", k, n)))
		}
	}
	if paged > 0 {
		parts = append(parts, sDim().Render(fmt.Sprintf("□ paged %d", paged)))
	}
	return strings.Join(parts, "  ")
}

// deviceName shortens the tier's "0:<name> [backend]" label to fit.
func deviceName(s string, w int) string {
	if _, rest, ok := strings.Cut(s, ":"); ok {
		s = rest
	}
	return truncate(strings.TrimPrefix(s, "#0 "), w)
}

// cardFull names a device whose pool is all but used, or "" when none is.
func cardFull(a session.Allocation) string {
	for _, d := range a.Devices {
		if d.Limit > 0 && float64(d.Used) >= 0.97*float64(d.Limit) {
			return deviceName(d.Name, 40)
		}
	}
	return ""
}

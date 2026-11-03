package main

import (
	"fmt"
	"slices"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/jitllm/jitllm/common/session"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// The relocation demo's machine map: one column per place a block can run,
// the model's blocks as rows in the column they are in, each with a bar for
// its history. Everything here is a pure function of what the placement
// reports, so a fixture renders the same text every time.

// hostID is the host's column. A block on a device carries that device's id.
const hostID = ""

// mapBlock is one block as the placement reports it.
type mapBlock struct {
	Index int
	Dev   string // device id; hostID for the host
	// The block's history cost (BlockPlacement.history_per_position,
	// history_window, state_bytes).
	PerPos uint64
	Window int
	State  uint64
}

// history is the bytes this block's history holds at position pos: the same
// arithmetic the move prices its copy by.
func (b mapBlock) history(pos int) uint64 {
	n := pos
	if b.Window > 0 {
		n = min(n, b.Window)
	}
	return b.State + uint64(n)*b.PerPos
}

// blocksOf reads a placement's blocks. A block with no id on a device is
// placed under the first of the model's devices, as the server reports a
// single unnamed one.
func blocksOf(p *v1.Placement) []mapBlock {
	out := make([]mapBlock, 0, len(p.GetBlocks()))
	for _, b := range p.GetBlocks() {
		mb := mapBlock{
			Index:  int(b.GetIndex()),
			PerPos: b.GetHistoryPerPosition().GetBytes(),
			Window: int(b.GetHistoryWindow()),
			State:  b.GetStateBytes().GetBytes(),
		}
		if b.GetLocation() == v1.BlockLocation_BLOCK_LOCATION_DEVICE {
			mb.Dev = b.GetDeviceId()
		}
		out = append(out, mb)
	}
	return out
}

// columnsOf is the map's columns: the host, then every device the model was
// loaded with, then any device a block is on that the list did not name.
func columnsOf(devices []string, blocks []mapBlock) []string {
	cols := []string{hostID}
	for _, d := range devices {
		if d != hostID && !slices.Contains(cols, d) {
			cols = append(cols, d)
		}
	}
	for _, b := range blocks {
		if !slices.Contains(cols, b.Dev) {
			cols = append(cols, b.Dev)
		}
	}
	return cols
}

func columnLabel(id string) string {
	if id == hostID {
		return "CPU host"
	}
	return id
}

func columnColour(cols []string, id string) lipgloss.Color {
	if id == hostID {
		return cBlue
	}
	return colourOf(slices.Index(cols, id)-1, deviceColours())
}

// slide is a block's arrival in a new column, played over a few frames after
// a move so the blocks are seen to travel: the block stays as a ghost where it
// was until its turn in the wave, then lands highlighted.
type slide struct {
	From  string
	Start int // the frame the wave began
	Rank  int // the block's place in the wave
}

const (
	slideStagger = 2  // blocks that leave per frame
	slideFlash   = 10 // frames a block stays highlighted after it lands
)

// slideAge is how far into its slide a block is at frame f: negative while it
// still waits at the source.
func (s slide) age(f int) int { return f - s.Start - s.Rank/slideStagger }

// mapView is everything the map draws.
type mapView struct {
	Cols   []string
	Blocks []mapBlock
	Pos    int // live position of the history
	MaxSeq int // the capacity the bars are drawn against
	Frame  int
	Slides map[int]slide // by block index
}

// blockRow is one line of a column.
func (v mapView) blockRow(b mapBlock, w int, col lipgloss.Color, ghost, landed bool) string {
	hist := b.history(v.Pos)
	full := b.history(v.MaxSeq)
	frac := 0.0
	if full > 0 {
		frac = float64(hist) / float64(full)
	}
	size := session.Bytes(hist)
	label := fmt.Sprintf("%02d", b.Index)
	barW := max(4, w-lipgloss.Width(label)-lipgloss.Width(size)-2)
	if ghost {
		return sDim().Render(label + " " + strings.Repeat("┄", barW) + " ⇢")
	}
	st := lipgloss.NewStyle().Foreground(col)
	mark := " "
	if landed {
		st = st.Bold(true)
		label = lipgloss.NewStyle().Foreground(cBrand).Bold(true).Render(label)
		mark = lipgloss.NewStyle().Foreground(cBrand).Bold(true).Render("◂")
	} else {
		label = sDim().Render(label)
	}
	// The mark takes the separator's place so the row never changes width.
	return label + mark + bar(frac, barW, st) + " " + sMuted().Render(size)
}

// bar is a w-cell bar at frac, in half cells so it grows a step at a time as a
// history does; the empty part is a faint rule.
func bar(frac float64, w int, fill lipgloss.Style) string {
	halves := int(min(max(frac, 0), 1)*float64(w*2) + 0.5)
	whole := halves / 2
	used := whole
	b := fill.Render(strings.Repeat("━", whole))
	if halves%2 == 1 && whole < w {
		b += fill.Render("╸")
		used++
	}
	return b + lipgloss.NewStyle().Foreground(cLine).Render(strings.Repeat("─", w-used))
}

// column draws one place and the blocks in it: w x h cells with the border.
func (v mapView) column(id string, w, h int) string {
	col := columnColour(v.Cols, id)
	iw, _ := inner(w, h)
	var rows []string
	var hist uint64
	n := 0
	for _, b := range v.Blocks {
		if b.Dev != id && !v.ghostedAt(b, id) {
			continue
		}
		sl, sliding := v.Slides[b.Index]
		age := 0
		if sliding {
			age = sl.age(v.Frame)
		}
		switch {
		case b.Dev == id && sliding && age < 0:
			continue // waiting at its source
		case b.Dev != id:
			rows = append(rows, v.blockRow(b, iw, col, true, false))
		default:
			landed := sliding && age < slideFlash
			rows = append(rows, v.blockRow(b, iw, col, false, landed))
			hist += b.history(v.Pos)
			n++
		}
	}
	head := lipgloss.NewStyle().Foreground(col).Bold(true).Render(fmt.Sprintf("%d blocks", n)) +
		sDim().Render(" · "+session.Bytes(hist))
	body := head + "\n\n" + strings.Join(rows, "\n")
	if len(rows) == 0 {
		body = head + "\n\n" + sDim().Render("· no blocks here ·")
	}
	return panel(columnLabel(id), body, w, h, col)
}

// ghostedAt is whether block b, already placed elsewhere, is still drawn in
// column id because its slide has not reached it.
func (v mapView) ghostedAt(b mapBlock, id string) bool {
	sl, ok := v.Slides[b.Index]
	return ok && sl.From == id && sl.age(v.Frame) < 0
}

// render is the whole map in w x h cells: the columns side by side.
func (v mapView) render(w, h int) string {
	n := len(v.Cols)
	cw := w / n
	cols := make([]string, n)
	for i, id := range v.Cols {
		width := cw
		if i == n-1 {
			width = w - cw*(n-1)
		}
		cols[i] = v.column(id, width, h)
	}
	return lipgloss.JoinHorizontal(lipgloss.Top, cols...)
}

package main

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/key"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/jitllm/jitllm/common/session"
)

type engineKeys struct{ In, Out, AllHost, AllDevice, NextDevice key.Binding }

// enginePage is the cockpit: every block and where it runs, the seam moved
// with the arrows, the rates over time, memory and the pager.
type enginePage struct {
	env  *env
	keys engineKeys
	w, h int
	// seam is the device block count asked for, so held arrows accumulate
	// before the engine answers; asked is true until the engine's own count
	// matches it or the engine is idle and has answered.
	seam  int
	asked bool
}

func newEnginePage(env *env) *enginePage {
	return &enginePage{env: env, keys: engineKeys{
		In:        bind("→", "block to device", "right", "l", "]"),
		Out:       bind("←", "block to host", "left", "h", "["),
		AllHost:   bind("home", "all host", "home"),
		AllDevice: bind("end", "all device", "end"),
		// The device-to-device relocation: every device block, its history
		// with it, onto the tier's next device.
		NextDevice: bind("d", "next device", "d"),
	}}
}

func (p *enginePage) Help() []key.Binding {
	return []key.Binding{p.keys.Out, p.keys.In, p.keys.AllHost, p.keys.AllDevice, p.keys.NextDevice}
}

func (p *enginePage) Focus() tea.Cmd   { return nil }
func (p *enginePage) Blur()            {}
func (p *enginePage) Capturing() bool  { return false }
func (p *enginePage) SetSize(w, h int) { p.w, p.h = w, h }
func (p *enginePage) Animating() bool {
	return p.env.tele.flashing(p.env.frame) || p.env.f.st.Streaming.Get() || p.asked
}

// ask moves the seam to n blocks.
func (p *enginePage) ask(n int) tea.Cmd {
	bm := p.env.f.st.BlockMap.Get()
	if len(bm) == 0 {
		return notify("load a model to move its blocks")
	}
	n = min(max(n, 0), len(bm))
	p.seam, p.asked = n, true
	p.env.e.Relocate(n)
	return nil
}

func (p *enginePage) Update(msg tea.Msg) tea.Cmd {
	st := p.env.f.st
	switch msg := msg.(type) {
	case seamMsg:
		return p.ask(msg.n)
	case engineMsg:
		// The engine moves the seam between jobs. The answer is in when the
		// count matches, or when the relocation reports that it could not
		// move it all ("... placed, n asked for", "the seam did not move").
		if p.asked && !st.Busy.Get() {
			status := st.Status.Get()
			if devBlocks(st.BlockMap.Get()) == p.seam ||
				strings.Contains(status, "asked for") || strings.Contains(status, "did not move") {
				p.asked = false
			}
		}
		if !p.asked {
			p.seam = devBlocks(st.BlockMap.Get())
		}
	case tea.KeyMsg:
		n := len(st.BlockMap.Get())
		switch {
		case key.Matches(msg, p.keys.In):
			return p.ask(p.seam + 1)
		case key.Matches(msg, p.keys.Out):
			return p.ask(p.seam - 1)
		case key.Matches(msg, p.keys.AllHost):
			return p.ask(0)
		case key.Matches(msg, p.keys.AllDevice):
			return p.ask(n)
		case key.Matches(msg, p.keys.NextDevice):
			if devBlocks(st.BlockMap.Get()) == 0 {
				return notify("no block is on a device to move")
			}
			p.env.e.MoveToNextDevice()
		}
	}
	return nil
}

func (p *enginePage) View() string {
	st := p.env.f.st
	bm := st.BlockMap.Get()
	if len(bm) == 0 {
		iw, ih := inner(p.w, p.h)
		msg := lipgloss.JoinVertical(lipgloss.Center,
			sBrand().Render("no model on the engine"), "",
			sDim().Render("f2 picks one; this screen then shows every block, where it runs,"),
			sDim().Render("and moves the seam between the host and the card with ← →"))
		return panel("engine", lipgloss.Place(iw, ih, lipgloss.Center, lipgloss.Center, msg), p.w, p.h, cBrand)
	}

	gw, _ := inner(p.w, 0)
	nd := devBlocks(bm)
	seam := sBold().Render(fmt.Sprint(nd)) + sDim().Render(fmt.Sprintf(" of %d blocks on a device", len(bm)))
	switch full := cardFull(st.Alloc.Get()); {
	case p.asked && p.seam != nd:
		when := ""
		if st.Busy.Get() {
			when = " after this reply"
		}
		seam += "  " + p.env.spin.View() + sMuted().Render(fmt.Sprintf(" moving to %d%s", p.seam, when))
	case full != "" && nd < len(bm):
		seam += "  " + lipgloss.NewStyle().Foreground(cOrange).Render("● "+full+" is full")
	}
	ctrl := sDim().Render("← → one block   home all host   end all device")
	line := seam + strings.Repeat(" ", max(2, gw-lipgloss.Width(seam)-lipgloss.Width(ctrl))) + ctrl
	d := nd * gw / len(bm)
	split := lipgloss.NewStyle().Foreground(cCyan).Render(strings.Repeat("▀", d)) +
		lipgloss.NewStyle().Foreground(cBrand).Render(strings.Repeat("▀", gw-d))
	_, name := activeModel(p.env)
	top := panel("blocks · "+strings.TrimSuffix(name, ".jlm"),
		lipgloss.JoinVertical(lipgloss.Left, p.env.tele.grid(bm, gw, true, p.env.frame), "", split, line, blockLegend(bm)),
		p.w, 0, cBrand)

	rest := max(8, p.h-lipgloss.Height(top))
	colW := p.w / 3
	lastW := p.w - 2*colW
	row := lipgloss.JoinHorizontal(lipgloss.Top,
		panel("throughput", p.throughput(colW, rest), colW, rest, cBlue),
		panel("memory", p.memory(colW, rest), colW, rest, cMagenta),
		panel("pager", p.pager(lastW, rest), lastW, rest, cOrange))
	return lipgloss.JoinVertical(lipgloss.Left, top, row)
}

func (p *enginePage) throughput(w, h int) string {
	st := p.env.f.st
	iw, ih := inner(w, h)
	var b strings.Builder
	last, peak, mean := p.env.tele.decode.stats()
	lines := 0
	if last > 0 {
		b.WriteString(sBrand().Render(fmt.Sprintf("%.1f", last)) + sDim().Render(" tok/s decode") + "\n")
		b.WriteString(kvLine("peak", fmt.Sprintf("%.1f", peak), iw) + "\n")
		b.WriteString(kvLine("mean", fmt.Sprintf("%.1f", mean), iw) + "\n")
		lines = 3
	} else {
		b.WriteString(sDim().Render("send a prompt to see it run") + "\n")
		lines = 1
	}
	var tail strings.Builder
	if r := st.PromptTokS.Get(); r > 0 {
		tail.WriteString("\n" + kvLine("prompt", fmt.Sprintf("%.0f tok/s", r), iw))
	}
	if g := st.GBs.Get(); g >= 0.05 {
		tail.WriteString("\n" + kvLine("bandwidth", fmt.Sprintf("%.1f GB/s", g), iw))
	}
	if bt := st.BytesPerTok.Get(); bt > 0 {
		tail.WriteString("\n" + kvLine("per token", session.Bytes(uint64(bt)), iw))
	}
	chartH := ih - lines - lipgloss.Height(tail.String()) - 2
	if ch := p.env.tele.decode.chart(iw, chartH, cBrand); ch != "" {
		b.WriteString("\n" + ch + "\n" + sDim().Render(fmt.Sprintf("last %ds", len(p.env.tele.decode.v)/2)))
	}
	return b.String() + tail.String()
}

func (p *enginePage) memory(w, h int) string {
	a := p.env.f.st.Alloc.Get()
	iw, _ := inner(w, h)
	var b strings.Builder
	if a.HostBudget > 0 {
		b.WriteString(kvLine("host pages", session.Bytes(a.HostUsed)+" / "+session.Bytes(a.HostBudget), iw) + "\n")
		b.WriteString(meter(float64(a.HostUsed)/float64(a.HostBudget), iw, cBrand, cBlue) + "\n")
	}
	if a.Dense > 0 {
		b.WriteString(kvLine("dense", session.Bytes(a.Dense), iw) + "\n")
	}
	if a.HostKV > 0 {
		b.WriteString(kvLine("host kv", session.Bytes(a.HostKV), iw) + "\n")
	}
	if a.PageBytes > 0 {
		b.WriteString(kvLine("page", session.Bytes(a.PageBytes), iw) + "\n")
	}
	for i, d := range a.Devices {
		c := colourOf(i, deviceColours())
		b.WriteString("\n" + lipgloss.NewStyle().Foreground(c).Bold(true).Render(deviceName(d.Name, iw)) + "\n")
		v := session.Bytes(d.Used)
		if d.Limit > 0 {
			v += " / " + session.Bytes(d.Limit)
		}
		b.WriteString(kvLine(fmt.Sprintf("%d blocks", d.Blocks), v, iw) + "\n")
		if d.Limit > 0 {
			b.WriteString(meter(float64(d.Used)/float64(d.Limit), iw, c, lerp(c, "#ffffff", 0.3)) + "\n")
		}
	}
	if a.MaxSeq > 0 {
		b.WriteString("\n" + kvLine("context", fmt.Sprintf("%d / %d", a.Pos, a.MaxSeq), iw) + "\n")
		b.WriteString(meter(float64(a.Pos)/float64(a.MaxSeq), iw, cMagenta, cBlue))
	}
	return b.String()
}

func (p *enginePage) pager(w, h int) string {
	st := p.env.f.st
	pg := st.Pager.Get()
	iw, _ := inner(w, h)
	var b strings.Builder
	b.WriteString(kvLine("frames held", fmt.Sprint(pg.Frames), iw) + "\n")
	b.WriteString(kvLine("resident", session.Bytes(pg.Resident), iw) + "\n")
	b.WriteString(kvLine("page-ins", fmt.Sprint(pg.PageIns), iw) + "\n")
	b.WriteString(kvLine("page-outs", fmt.Sprint(pg.PageOuts), iw) + "\n")
	b.WriteString(kvLine("read", fmt.Sprintf("%s in %d reads", session.Bytes(pg.BytesRead), pg.Reads), iw) + "\n")
	switch {
	case pg.TurnOuts > 0:
		b.WriteString(lipgloss.NewStyle().Foreground(cOrange).Render(fmt.Sprintf("paging: %d evictions this turn", pg.TurnOuts)) + "\n")
	case pg.Frames > 0:
		b.WriteString(sBrand().Render("no evictions this turn") + "\n")
	}
	if s := st.ModelSummary.Get(); s != "" {
		b.WriteString("\n" + lipgloss.NewStyle().Foreground(cMuted).Width(iw).Render(s) + "\n")
	}
	if s := st.DeviceReport.Get(); s != "" {
		b.WriteString("\n" + lipgloss.NewStyle().Foreground(cDim).Width(iw).Render(s))
	}
	return b.String()
}

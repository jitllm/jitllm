package main

import (
	"fmt"
	"io"
	"strings"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/list"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/jitllm/jitllm/common/discover"
	"github.com/jitllm/jitllm/common/hardware"
	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/convert/library"
)

type libItem struct {
	library.Model
	i int
}

func (it libItem) FilterValue() string { return it.Title + " " + it.Name + " " + it.Arch }

type libRow struct{ p *discoverPage }

func (libRow) Height() int                         { return 1 }
func (libRow) Spacing() int                        { return 0 }
func (libRow) Update(tea.Msg, *list.Model) tea.Cmd { return nil }
func (c libRow) Render(w io.Writer, l list.Model, i int, li list.Item) {
	it := li.(libItem)
	p := c.p
	marker, ns := "  ", sFg
	if i == l.Index() {
		marker, ns = sBrand.Render("▌ "), sBold
	}
	fit := ""
	if p.onDisk(it.i) {
		fit = sBrand.Render("✓ on disk")
	} else if p.probed {
		fit = fitText(discover.FitWords(it.Model, p.mr))
	}
	right := fit + strings.Repeat(" ", max(1, 20-lipgloss.Width(fit))) +
		sMuted.Render(fmt.Sprintf("%-8s%-7s", it.Params, it.Quant)) + sDim.Render(fmt.Sprintf("%10s", session.Bytes(uint64(it.Bytes))))
	title := it.Title
	if it.MMProj != "" {
		title += " ◉"
	}
	room := l.Width() - 2 - lipgloss.Width(right) - 1
	title = truncate(title, room)
	fmt.Fprint(w, marker+ns.Render(title)+strings.Repeat(" ", max(1, room-lipgloss.Width(title)+1))+right)
}

// fitText colours where a model would live: green on the card or in memory,
// orange where it pages, red where it does not fit.
func fitText(fit string) string {
	col := cBrand
	switch l := strings.ToLower(fit); {
	case strings.Contains(l, "not"), strings.Contains(l, "too"):
		col = cRed
	case strings.Contains(l, "page"), strings.Contains(l, "slow"):
		col = cOrange
	}
	return lipgloss.NewStyle().Foreground(col).Render(fit)
}

// machineMsg is the hardware probe, taken on the engine's queue.
type machineMsg struct{ mr session.MachineReport }

type discoverKeys struct{ Get, Filter, Smaller, Larger, Stop key.Binding }

// discoverPage is the models jitllm has verified, ranked for this machine by
// the window's ranking (discover.Rank).
type discoverPage struct {
	env    *env
	keys   discoverKeys
	list   list.Model
	mr     session.MachineReport
	probed bool
	disk   []bool
	pref   int
	w, h   int
	listW  int
}

func newDiscoverPage(env *env) *discoverPage {
	p := &discoverPage{env: env, pref: 2, keys: discoverKeys{
		Get:     bind("enter", "download · load"),
		Filter:  bind("type", "filter", "/"),
		Smaller: bind("<", "faster", "<", ","),
		Larger:  bind(">", "smarter", ">", "."),
		Stop:    bind("ctrl+x", "stop download"),
	}}
	p.list = list.New(nil, libRow{p}, 60, 20)
	p.list.SetShowPagination(false)
	p.list.SetShowTitle(false)
	p.list.SetShowHelp(false)
	p.list.DisableQuitKeybindings()
	p.list.FilterInput.Prompt = sBrand.Render("  ")
	p.list.Styles.StatusBar = sDim.PaddingLeft(2).PaddingBottom(1)
	p.refresh()
	return p
}

// probeMachine measures the machine on the engine's queue, so it never
// contends with a device a model holds.
func probeMachine(env *env) {
	env.e.Run(func() { env.f.queue <- machineMsg{hardware.Probe().Machine()} })
}

func (p *discoverPage) onDisk(i int) bool { return i < len(p.disk) && p.disk[i] }

// refresh ranks the library for the machine and checks what is on disk.
func (p *discoverPage) refresh() {
	p.disk = discover.OnDisk(discover.Dir(p.env.dirs))
	order := make([]int, len(library.Models()))
	for i := range order {
		order[i] = i
	}
	if p.probed {
		order = discover.Rank(library.Models(), p.mr, p.pref)
	}
	items := make([]list.Item, 0, len(order))
	for _, i := range order {
		items = append(items, libItem{library.Models()[i], i})
	}
	p.list.SetItems(items)
}

func (p *discoverPage) Help() []key.Binding {
	return []key.Binding{p.keys.Get, p.keys.Filter, p.keys.Smaller, p.keys.Larger, p.keys.Stop}
}

func (p *discoverPage) Focus() tea.Cmd  { return nil }
func (p *discoverPage) Blur()           {}
func (p *discoverPage) Capturing() bool { return p.list.FilterState() != list.Unfiltered }
func (p *discoverPage) Animating() bool { return !p.probed || p.env.jobs.busy() }

func (p *discoverPage) SetSize(w, h int) {
	p.w, p.h = w, h
	p.listW = w * 3 / 5
	iw, ih := inner(p.listW, h)
	p.list.SetSize(iw, ih-2)
}

func (p *discoverPage) Update(msg tea.Msg) tea.Cmd {
	switch msg := msg.(type) {
	case machineMsg:
		p.mr, p.probed = msg.mr, true
		p.refresh()
		return nil
	case rescanMsg:
		p.refresh()
		return nil
	case tea.KeyMsg:
		// Enter on a filter keeps it and opens what it found, one key.
		var kept tea.Cmd
		if p.list.FilterState() == list.Filtering && key.Matches(msg, p.keys.Get) {
			p.list, kept = p.list.Update(msg)
		}
		if key.Matches(msg, p.keys.Stop) {
			p.env.jobs.cancel()
			return kept
		}
		if p.list.FilterState() != list.Filtering {
			switch {
			case key.Matches(msg, p.keys.Get):
				it, ok := p.list.SelectedItem().(libItem)
				if !ok {
					return kept
				}
				if p.onDisk(it.i) {
					return tea.Batch(kept, load(discover.ContainerFor(discover.Dir(p.env.dirs), it.Model)))
				}
				return tea.Batch(kept, p.env.jobs.download(it.Model))
			case key.Matches(msg, p.keys.Smaller):
				p.pref = max(0, p.pref-1)
				p.refresh()
				return nil
			case key.Matches(msg, p.keys.Larger):
				p.pref = min(len(discover.BalanceSteps)-1, p.pref+1)
				p.refresh()
				return nil
			}
		}
	}
	var pre tea.Cmd
	if typing(msg, p.list) && !key.Matches(msg.(tea.KeyMsg), p.keys.Smaller, p.keys.Larger) {
		p.list, pre = p.list.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'/'}})
	}
	var cmd tea.Cmd
	p.list, cmd = p.list.Update(msg)
	return tea.Batch(pre, cmd)
}

func (p *discoverPage) View() string {
	var steps []string
	for i, s := range discover.BalanceSteps {
		if i == p.pref {
			steps = append(steps, pill(s, cBg, cBrand))
		} else {
			steps = append(steps, sDim.Render(" "+s+" "))
		}
	}
	bal := sDim.Render("< ") + strings.Join(steps, "") + sDim.Render(" >")
	if !p.probed {
		bal = p.env.spin.View() + sDim.Render(" measuring this machine to rank them…")
	}
	left := panel(fmt.Sprintf("discover · %d verified models", len(library.Models())),
		bal+"\n\n"+p.list.View(), p.listW, p.h, cBrand)

	rw, _ := inner(p.w-p.listW, p.h)
	var b strings.Builder
	if it, ok := p.list.SelectedItem().(libItem); ok {
		b.WriteString(sBold.Render(truncate(it.Title, rw)) + "\n")
		b.WriteString(sDim.Render(truncate(it.Repo, rw)) + "\n\n")
		row := func(k, v string) {
			if v != "" {
				b.WriteString(kvLine(k, v, rw) + "\n")
			}
		}
		row("arch", it.Arch)
		row("params", it.Params)
		row("quant", it.Quant)
		row("download", session.Bytes(uint64(it.Bytes)))
		if p.probed {
			row("fits", discover.FitWords(it.Model, p.mr))
			row("speed", discover.Speed(it.Model, p.mr))
		}
		if it.Note != "" {
			b.WriteString("\n" + lipgloss.NewStyle().Foreground(cMuted).Width(rw).Render(it.Note) + "\n")
		}
		if it.Verified != "" {
			b.WriteString("\n" + sDim.Render(truncate("verified "+it.Verified, rw)) + "\n")
		}
		b.WriteString("\n")
		switch note := discover.SpaceNote(it.Model, discover.Dir(p.env.dirs), discover.FreeBytes); {
		case p.onDisk(it.i):
			b.WriteString(sBrand.Render("enter loads it"))
		case note != "":
			b.WriteString(lipgloss.NewStyle().Foreground(cRed).Width(rw).Render(note))
		default:
			b.WriteString(sMuted.Render("enter downloads it and converts it to a .jlm"))
		}
	}
	if jv := p.env.jobs.View(rw); jv != "" {
		b.WriteString("\n\n" + jv)
	}
	return lipgloss.JoinHorizontal(lipgloss.Top, left, panel("details", b.String(), p.w-p.listW, p.h, cBlue))
}

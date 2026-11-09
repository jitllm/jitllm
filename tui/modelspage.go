package main

import (
	"fmt"
	"io"
	"strings"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/list"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/jitllm/jitllm/common/catalog"
	"github.com/jitllm/jitllm/common/convertjob"
	"github.com/jitllm/jitllm/common/session"
)

// item is one catalog entry in the list.
type item struct{ catalog.Entry }

func (i item) FilterValue() string { return i.Name }

// row draws an entry as one aligned line: its name, what it is, its quant
// and size, and whether the engine has it open.
type row struct{ env *env }

func (row) Height() int                         { return 1 }
func (row) Spacing() int                        { return 0 }
func (row) Update(tea.Msg, *list.Model) tea.Cmd { return nil }
func (r row) Render(w io.Writer, l list.Model, i int, li list.Item) {
	it := li.(item)
	sel := i == l.Index()
	marker, ns := "  ", sFg()
	if sel {
		marker, ns = sBrand().Render("▌ "), sBold()
	}
	tag := ""
	switch {
	case it.Kind == catalog.KindGGUF:
		tag = lipgloss.NewStyle().Foreground(cOrange).Render("gguf · convert")
	case it.Kind == catalog.KindSafetensors:
		tag = lipgloss.NewStyle().Foreground(cMagenta).Render("safetensors · convert")
	case strings.Contains(it.Name, "vlm") || it.Tower:
		tag = lipgloss.NewStyle().Foreground(cCyan).Render("vision")
	}
	for _, lm := range r.env.f.st.Models.Get() {
		if lm.Path == it.Path {
			dot := lipgloss.NewStyle().Foreground(colourOf(lm.Colour, modelColours()))
			tag = dot.Render("● " + strings.ToLower(session.Loaded))
		}
	}
	quant := catalog.QuantFromName(it.Name)
	right := tag + strings.Repeat(" ", max(1, 22-lipgloss.Width(tag))) +
		sMuted().Render(fmt.Sprintf("%-7s", quant)) + sDim().Render(fmt.Sprintf("%10s", session.Bytes(uint64(it.Size))))
	name := strings.TrimSuffix(strings.TrimSuffix(it.Name, ".jlm"), ".gguf")
	room := l.Width() - 2 - lipgloss.Width(right) - 1
	name = truncate(name, room)
	fmt.Fprint(w, marker+ns.Render(name)+strings.Repeat(" ", max(1, room-lipgloss.Width(name)+1))+right)
}

// probedMsg carries an entry whose header was read.
type probedMsg struct{ e catalog.Entry }

type modelsKeys struct{ Load, Close, Filter key.Binding }

// modelsPage is every model in the configured folders: open one, switch to
// it, unload it, or convert a source into a container.
type modelsPage struct {
	env  *env
	keys modelsKeys
	list list.Model
	// probed holds each file's header once read; probing the ones being read.
	probed  map[string]catalog.Entry
	probing map[string]bool
	w, h    int
	listW   int
}

func newModelsPage(env *env) *modelsPage {
	l := list.New(nil, row{env}, 60, 20)
	l.SetShowTitle(false)
	l.SetShowHelp(false)
	l.DisableQuitKeybindings()
	l.FilterInput.Prompt = sBrand().Render("  ")
	l.FilterInput.Cursor.Style = sBrand()
	l.Styles.StatusBar = sDim().PaddingLeft(2).PaddingBottom(1)
	l.SetShowPagination(false)
	l.Styles.NoItems = sDim().PaddingLeft(2)
	p := &modelsPage{
		env:     env,
		keys:    modelsKeys{Load: bind("enter", "load · convert"), Close: bind("ctrl+x", "unload model"), Filter: bind("type", "filter", "/")},
		list:    l,
		probed:  map[string]catalog.Entry{},
		probing: map[string]bool{},
	}
	p.scan()
	return p
}

// scan lists the model folders, containers first: they are what the engine
// runs.
func (p *modelsPage) scan() {
	var runs, sources []list.Item
	for _, e := range catalog.Scan(p.env.dirs) {
		switch e.Kind {
		case catalog.KindContainer:
			runs = append(runs, item{e})
		case catalog.KindGGUF, catalog.KindSafetensors:
			sources = append(sources, item{e})
		}
	}
	p.list.SetItems(append(runs, sources...))
	// A rewritten file is not the file whose header was read.
	p.probed = map[string]catalog.Entry{}
}

func (p *modelsPage) Help() []key.Binding {
	return []key.Binding{p.keys.Load, p.keys.Filter, bind("↑↓", "move"), p.keys.Close}
}

func (p *modelsPage) Focus() tea.Cmd  { return p.probeSelected() }
func (p *modelsPage) Blur()           {}
func (p *modelsPage) Capturing() bool { return p.list.FilterState() != list.Unfiltered }
func (p *modelsPage) Animating() bool { return p.env.jobs.busy() }

func (p *modelsPage) SetSize(w, h int) {
	p.w, p.h = w, h
	p.listW = w * 3 / 5
	iw, ih := inner(p.listW, h)
	p.list.SetSize(iw, ih)
}

func (p *modelsPage) probeSelected() tea.Cmd {
	it, ok := p.list.SelectedItem().(item)
	if !ok {
		return nil
	}
	if _, done := p.probed[it.Path]; done || p.probing[it.Path] {
		return nil
	}
	p.probing[it.Path] = true
	e := it.Entry
	return func() tea.Msg {
		catalog.Probe(&e)
		return probedMsg{e}
	}
}

func (p *modelsPage) Update(msg tea.Msg) tea.Cmd {
	switch msg := msg.(type) {
	case probedMsg:
		p.probed[msg.e.Path] = msg.e
		delete(p.probing, msg.e.Path)
		return nil
	case rescanMsg:
		p.scan()
		return p.probeSelected()
	case tea.KeyMsg:
		// Enter on a filter keeps it and opens what it found, one key.
		var kept tea.Cmd
		if p.list.FilterState() == list.Filtering && key.Matches(msg, p.keys.Load) {
			p.list, kept = p.list.Update(msg)
		}
		if p.list.FilterState() != list.Filtering {
			if it, ok := p.list.SelectedItem().(item); ok {
				switch {
				case key.Matches(msg, p.keys.Load):
					return tea.Batch(kept, p.open(it))
				case key.Matches(msg, p.keys.Close):
					if isOpen(p.env, it.Path) {
						p.env.e.Unload(it.Path)
						return notify("unloaded " + it.Name)
					}
					return nil
				}
			}
		}
	}
	var pre tea.Cmd
	if typing(msg, p.list) {
		// Typing a name filters at once, without the list's "/" first.
		p.list, pre = p.list.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'/'}})
	}
	var cmd tea.Cmd
	p.list, cmd = p.list.Update(msg)
	return tea.Batch(pre, cmd, p.probeSelected())
}

// typing is a printable key on a list that is not filtering yet.
func typing(msg tea.Msg, l list.Model) bool {
	k, ok := msg.(tea.KeyMsg)
	return ok && k.Type == tea.KeyRunes && !k.Alt && l.FilterState() != list.Filtering
}

// open loads a container, switches to it when it is open, converts a
// source, or rebuilds an outdated container as the window's Convert tab does.
func (p *modelsPage) open(it item) tea.Cmd {
	if it.Kind != catalog.KindContainer {
		return p.env.jobs.convert(convertjob.Request{Src: it.Path, Dir: convertjob.PrimaryDir(p.env.dirs)}, it.Name)
	}
	if e, ok := p.probed[it.Path]; ok && e.Stale {
		r := convertjob.PlanReconvert(it.Path, p.env.jobs.busy())
		if !r.Start {
			return notify(r.Note)
		}
		return p.env.jobs.convert(r.Req, it.Name)
	}
	return load(it.Path)
}

func (p *modelsPage) View() string {
	title := fmt.Sprintf("models · %d", len(p.list.Items()))
	if n := len(p.env.f.st.Models.Get()); n > 0 {
		title += fmt.Sprintf(" · %d loaded", n)
	}
	left := panel(title, p.list.View(), p.listW, p.h, cBrand)
	rw, _ := inner(p.w-p.listW, p.h)
	var info string
	if it, ok := p.list.SelectedItem().(item); ok {
		e, probed := p.probed[it.Path]
		if !probed {
			e = it.Entry
		}
		info = details(e, probed, rw)
	} else {
		info = sDim().Render("no models found.\n\nadd a folder in Settings (f5), set JITLLM_MODELS,\nor pass -models dir")
	}
	if jv := p.env.jobs.View(rw); jv != "" {
		info += "\n\n" + jv
	}
	return lipgloss.JoinHorizontal(lipgloss.Top, left, panel("details", info, p.w-p.listW, p.h, cBlue))
}

// details is what the selected file is.
func details(e catalog.Entry, probed bool, w int) string {
	var b strings.Builder
	b.WriteString(sBold().Render(truncate(e.Name, w)) + "\n")
	b.WriteString(sDim().Render(truncate(e.Path, w)) + "\n\n")
	row := func(k, v string) { b.WriteString(kvLine(k, v, w) + "\n") }
	switch {
	case !probed:
		b.WriteString(sDim().Render("reading header…"))
	case e.ProbeErr != "":
		b.WriteString(lipgloss.NewStyle().Foreground(cRed).Width(w).Render(e.ProbeErr))
	default:
		row("blocks", fmt.Sprint(e.NLayer))
		row("width", fmt.Sprint(e.NEmbd))
		row("context", fmt.Sprint(e.NCtx))
		row("vocab", fmt.Sprint(e.NVocab))
		if e.MoE() {
			row("experts", fmt.Sprintf("%d of %d", e.NExpertUsed, e.NExpert))
		}
		if e.Quant != "" {
			row("quant", e.Quant)
		}
		if e.PageSize > 0 {
			row("page", session.Bytes(e.PageSize))
		}
		var caps []string
		if e.HasChat {
			caps = append(caps, chip("chat template", cBrand))
		}
		if e.HasVision {
			caps = append(caps, chip(fmt.Sprintf("vision · %d blocks", e.VisionBlocks), cCyan))
		}
		if e.Stale {
			caps = append(caps, chip("outdated container", cRed))
		}
		if len(caps) > 0 {
			b.WriteString("\n" + strings.Join(caps, " "))
		}
	}
	switch {
	case e.Kind != catalog.KindContainer:
		b.WriteString("\n\n" + sMuted().Render("enter converts it to a .jlm, the format the engine runs"))
	case e.Stale:
		b.WriteString("\n\n" + sMuted().Render("enter rebuilds it for this version"))
	default:
		b.WriteString("\n\n" + sDim().Render("enter loads it · ctrl+x unloads it"))
	}
	return b.String()
}

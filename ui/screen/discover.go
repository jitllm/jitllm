package screen

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/gogpu/ui/core/button"
	"github.com/gogpu/ui/core/chip"
	"github.com/gogpu/ui/core/progressbar"
	"github.com/gogpu/ui/core/scrollview"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/common/discover"
	"github.com/jitllm/jitllm/common/hardware"
	"github.com/jitllm/jitllm/convert/library"
	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/widgets"
)

// Discover is the first thing a person needs: what this machine is, and which
// models it runs well, ranked for it, one press from the Hub to a loaded
// model. The download itself is downloadScreen's (download.go).
func Discover(sh *app.Shell) widget.Widget {
	StartProbe(sh)
	d := downloadsFor(sh)
	p := sh.P
	// The panel shows the model the ranking puts first until a person picks
	// one: the probe and the balance both reorder the list under it.
	follow := func([]int) {
		if o := d.order().Get(); !d.picked && len(o) > 0 {
			d.sel.Set(o[0])
		}
	}
	follow(nil)

	title := p.Role(primitives.Text("Discover").Color(p.Text()), p.Type.HeadlineSmall).Bold()
	sub := widgets.NewParagraph("Models this machine runs, ranked for it. Each one converts once into a .jlm the first time you get it.").
		FontSize(p.Type.BodyMedium.FontSize).Color(p.Muted()).Font(app.Prose)

	body := primitives.VBox(
		primitives.VBox(title, sub).Gap(p.Space.XS).CrossAlign(primitives.CrossAxisStretch),
		machineCard(sh),
		d.balance(),
		primitives.HBox(
			primitives.Expanded(d.ranked()),
			primitives.Box(d.panel()).Width(discoverPanelWidth),
		).Gap(p.Space.M),
	).Gap(p.Space.L).CrossAlign(primitives.CrossAxisStretch).Padding(p.Space.L)

	return primitives.Box(primitives.VBox(
		primitives.Expanded(scrollview.New(body, scrollview.PainterOpt(p.Scrollbar))),
		newSignalWatcher(d.order(), follow),
	)).Background(p.Background())
}

// discoverPanelWidth is the selected model's panel beside the list.
const discoverPanelWidth = 340

// --- the machine ---------------------------------------------------------------

// machineCard says what this machine is, in the terms that decide what it
// runs: cores, memory for weights, the bandwidth a token is read at, the GPU.
func machineCard(sh *app.Shell) widget.Widget {
	p := sh.P
	rep := func(f func(hardware.Report) string) state.ReadonlySignal[string] {
		return state.NewComputed(func() string {
			r, ok := depsOf(sh).Machine.Cached()
			if !ok {
				return "reading..."
			}
			return f(r)
		}, sh.Store.Machine.AsReadonly())
	}
	stat := func(label string, value, note state.ReadonlySignal[string]) widget.Widget {
		return primitives.VBox(
			p.Role(primitives.Text(label).Color(p.Muted()), p.Type.LabelMedium),
			p.Role(primitives.Text("").ContentSignal(value).Color(p.Text()), p.Type.TitleSmall).Bold(),
			primitives.Text("").ContentSignal(note).FontSize(p.Type.BodySmall.FontSize).Color(p.Muted()),
		).Gap(2)
	}
	gpu := func(r hardware.Report) (hardware.GPU, bool) {
		for _, g := range r.GPUs {
			if !g.Unified {
				return g, true
			}
		}
		return hardware.GPU{}, false
	}

	return primitives.Box(primitives.HBox(
		widgets.NewIconTile(widgets.IconMachine, 56, p.Colors.SurfaceContainerHigh, p.Colors.Primary),
		primitives.Expanded(primitives.VBox(
			p.Role(primitives.Text("This machine").Color(p.Muted()), p.Type.LabelMedium),
			p.Role(primitives.Text("").ContentSignal(rep(func(r hardware.Report) string { return r.CPU })).Color(p.Text()), p.Type.TitleMedium).Bold(),
			primitives.Box().Height(p.Space.XS),
			primitives.HBox(
				stat("Cores",
					rep(func(r hardware.Report) string {
						if len(r.ECores) > 0 {
							return fmt.Sprintf("%d P + %d E", len(r.PCores), len(r.ECores))
						}
						return fmt.Sprintf("%d", len(r.PCores))
					}),
					rep(func(r hardware.Report) string { return fmt.Sprintf("%d write the reply", len(r.Decode)) })),
				stat("Memory",
					rep(func(r hardware.Report) string { return app.Bytes(r.RAMTotal) }),
					rep(func(r hardware.Report) string { return app.Bytes(r.MemBudget) + " for weights" })),
				stat("Bandwidth",
					rep(func(r hardware.Report) string {
						if r.MemWall <= 0 {
							return "--"
						}
						return app.GBs(r.MemWall)
					}),
					rep(func(r hardware.Report) string { return "measured, decides decode speed" })),
				stat("GPU",
					rep(func(r hardware.Report) string {
						if g, ok := gpu(r); ok {
							return hardware.ShortGPU(g.Name)
						}
						return "none"
					}),
					rep(func(r hardware.Report) string {
						if g, ok := gpu(r); ok {
							return app.Bytes(g.Total) + " " + strings.ToUpper(g.API)
						}
						return "runs on the CPU"
					})),
			).Gap(p.Space.XL),
		).Gap(2).CrossAlign(primitives.CrossAxisStretch)),
	).Gap(p.Space.M)).
		Padding(p.Space.M+p.Space.XS).
		Rounded(p.Shape.Large).
		BorderStyle(1, p.Colors.OutlineVariant).
		Background(p.Colors.SurfaceContainer)
}

// --- the balance ---------------------------------------------------------------

func (d *downloadScreen) balance() widget.Widget {
	p := d.sh.P
	chips := make([]widget.Widget, len(discover.BalanceSteps))
	for i, s := range discover.BalanceSteps {
		chips[i] = chip.New(
			chip.Label(s),
			chip.Selectable(true),
			chip.SelectedReadonlySignal(state.NewComputed(func() bool { return d.pref.Get() == i }, d.pref.AsReadonly())),
			chip.OnClick(func() { d.pref.Set(i) }),
			chip.PainterOpt(p.Chip),
		)
	}
	return primitives.VBox(
		p.Role(primitives.Text("Find your balance").Color(p.Text()), p.Type.TitleSmall).Bold(),
		primitives.Text("Quick answers from a small model, or deeper ones from a large one.").
			FontSize(p.Type.BodySmall.FontSize).Color(p.Muted()),
		primitives.Box().Height(p.Space.XS),
		primitives.HBox(chips...).Gap(p.Space.S),
	).Gap(2)
}

// SetBalance sets Discover's preference, as its chips do: 0 is Fastest.
func SetBalance(sh *app.Shell, step int) {
	downloadsFor(sh).pref.Set(min(max(step, 0), len(discover.BalanceSteps)-1))
}

// ShowDownload puts Discover in the state a download in flight leaves it:
// busy, the bar at fraction, the note beside it. The mock scenarios paint it
// without a Hub.
func ShowDownload(sh *app.Shell, fraction float64, note string) {
	d := downloadsFor(sh)
	d.busy.Set(true)
	d.progress.Set(fraction)
	d.note.Set(note)
}

// --- the ranking -----------------------------------------------------------------

// order is the current ranking.
func (d *downloadScreen) order() state.ReadonlySignal[[]int] {
	if d.ord == nil {
		d.ord = state.NewComputed(func() []int {
			return discover.Rank(library.Models(), d.sh.Store.Machine.Get(), d.pref.Get())
		}, d.pref.AsReadonly(), d.sh.Store.Machine.AsReadonly())
	}
	return d.ord
}

// statusOf is a library model's row status: whether it is already here.
func (d *downloadScreen) statusOf(i int) string {
	if d.onDisk(i) {
		return "On this machine"
	}
	return ""
}

// ranked is one row per library model, in ranking order. A row is a slot --
// the model at rank k -- so a new preference re-points the rows rather than
// rebuilding them.
func (d *downloadScreen) ranked() widget.Widget {
	p := d.sh.P
	order := d.order()
	at := func(k int) int {
		o := order.Get()
		if k < len(o) {
			return o[k]
		}
		return -1
	}
	text := func(k int, f func(library.Model, int) string) state.ReadonlySignal[string] {
		return state.NewComputed(func() string {
			if i := at(k); i >= 0 {
				return f(library.Models()[i], i)
			}
			return ""
		}, order, d.have.AsReadonly(), d.sh.Store.Machine.AsReadonly())
	}
	palette := []widget.Color{p.Colors.Primary, p.Colors.Secondary, p.Colors.Tertiary, p.Warn}
	rows := make([]widget.Widget, len(library.Models()))
	for k := range library.Models() {
		look := state.NewComputed(func() widgets.TileLook {
			i := at(k)
			if i < 0 {
				return widgets.TileLook{}
			}
			fam := strings.Fields(library.Models()[i].Title)[0]
			h := 0
			for _, r := range fam {
				h += int(r)
			}
			return widgets.TileLook{Letter: strings.ToUpper(fam[:1]), Bg: palette[h%len(palette)]}
		}, order)
		row := primitives.Box(primitives.HBox(
			primitives.Box(primitives.Text(strconv.Itoa(k+1)).FontSize(p.Type.BodySmall.FontSize).Color(p.Muted())).Width(22).PaddingTop(10),
			widgets.NewLetterTile("", 36, p.Colors.Primary, p.Colors.OnPrimary).Bind(look),
			primitives.Expanded(primitives.VBox(
				p.Role(primitives.Text("").ContentSignal(text(k, func(m library.Model, _ int) string { return m.Title })).Color(p.Text()), p.Type.BodyLarge).Bold(),
				primitives.Text("").ContentSignal(text(k, func(m library.Model, i int) string {
					s := m.Params + " · " + m.Quant + " · " + app.Bytes(uint64(m.Bytes))
					if m.Vision() {
						s += " · sees images"
					}
					if st := d.statusOf(i); st != "" {
						s += " · " + st
					}
					return s
				})).FontSize(p.Type.BodySmall.FontSize).Color(p.Muted()),
			).Gap(2)),
			primitives.Box(primitives.VBox(
				widgets.NewParagraph("").ContentSignal(text(k, func(m library.Model, _ int) string {
					return discover.Speed(m, d.sh.Store.Machine.Get())
				})).FontSize(p.Type.BodySmall.FontSize).Color(p.Text()).Font(app.Mono).MaxLines(1).Align(widget.TextAlignRight),
				widgets.NewParagraph("").ContentSignal(text(k, func(m library.Model, _ int) string {
					return discover.FitWords(m, d.sh.Store.Machine.Get())
				})).FontSize(p.Type.BodySmall.FontSize).Color(p.Muted()).Font(app.Prose).MaxLines(1).Align(widget.TextAlignRight),
			).Gap(2)).Width(150),
		).Gap(p.Space.S+p.Space.XS)).PaddingXY(p.Space.S+p.Space.XS, p.Space.S+p.Space.XS)
		sel := state.NewComputed(func() bool { i := at(k); return i >= 0 && d.sel.Get() == i }, order, d.sel.AsReadonly())
		rows[k] = widgets.NewClickable(row, sel, func() {
			if i := at(k); i >= 0 {
				d.picked = true
				d.sel.Set(i)
			}
		}, p.Colors.SurfaceContainer, p.Colors.SurfaceContainerHigh, p.Colors.OutlineVariant)
	}
	return primitives.VBox(rows...).Gap(2).CrossAlign(primitives.CrossAxisStretch)
}

// --- the selected model ------------------------------------------------------------

func (d *downloadScreen) panel() widget.Widget {
	p := d.sh.P
	pick := func(f func(library.Model, int) string) state.ReadonlySignal[string] {
		return state.NewComputed(func() string {
			m, i, ok := d.selected()
			if !ok {
				return ""
			}
			return f(m, i)
		}, d.sel.AsReadonly(), d.have.AsReadonly(), d.sh.Store.Machine.AsReadonly())
	}
	metric := func(label string, value state.ReadonlySignal[string]) widget.Widget {
		return primitives.VBox(
			p.Role(primitives.Text(label).Color(p.Muted()), p.Type.LabelMedium),
			widgets.NewParagraph("").ContentSignal(value).FontSize(p.Type.BodyMedium.FontSize).Color(p.Text()).Font(app.Prose).Bold().MaxLines(2),
		).Gap(2)
	}

	action := button.New(
		button.TextReadonlySignal(pick(func(m library.Model, i int) string {
			if d.onDisk(i) {
				return "Load"
			}
			return "Download " + app.Bytes(uint64(m.Bytes))
		})),
		button.SizeOpt(button.Large),
		button.VariantOpt(button.Filled),
		button.DisabledReadonlySignal(state.NewComputed(func() bool {
			_, _, ok := d.selected()
			return !ok || d.busy.Get()
		}, d.sel.AsReadonly(), d.busy.AsReadonly())),
		button.OnClick(d.act),
		button.PainterOpt(p.Button),
	)
	bar := widgets.Hide(d.busy.AsReadonly(), primitives.HBox(
		primitives.Expanded(progressbar.New(
			progressbar.ValueReadonlySignal(d.progress.AsReadonly()),
			progressbar.ColorSchemeOpt(p.ProgressBar),
			progressbar.Height(6),
		)),
		button.New(
			button.TextOpt("Stop"),
			button.SizeOpt(button.Small),
			button.VariantOpt(button.Outlined),
			button.OnClick(d.cancel),
			button.PainterOpt(p.Button),
		),
	).Gap(p.Space.S).CrossAlign(primitives.CrossAxisCenter))
	none := state.NewComputed(func() bool { _, _, ok := d.selected(); return !ok }, d.sel.AsReadonly())
	some := state.NewComputed(func() bool { return !none.Get() }, none)

	details := widgets.Hide(some, primitives.VBox(
		p.Role(primitives.Text("").ContentSignal(pick(func(m library.Model, _ int) string { return m.Title })).Color(p.Text()), p.Type.TitleLarge).Bold(),
		widgets.NewParagraph("").ContentSignal(pick(func(m library.Model, _ int) string { return m.Note })).
			FontSize(p.Type.BodySmall.FontSize).Color(p.Muted()).Font(app.Prose),
		primitives.Box().Height(p.Space.XS),
		primitives.HBox(
			primitives.Expanded(metric("Speed", pick(func(m library.Model, _ int) string {
				if s := discover.Speed(m, d.sh.Store.Machine.Get()); s != "" {
					return s
				}
				return "--"
			}))),
			primitives.Expanded(metric("Fit", pick(func(m library.Model, _ int) string { return discover.FitWords(m, d.sh.Store.Machine.Get()) }))),
		).Gap(p.Space.M),
		primitives.HBox(
			primitives.Expanded(metric("Size", pick(func(m library.Model, _ int) string { return m.Params + " · " + m.Quant }))),
			primitives.Expanded(metric("Download", pick(func(m library.Model, _ int) string { return app.Bytes(uint64(m.Bytes)) }))),
		).Gap(p.Space.M),
		primitives.Box().Height(p.Space.XS),
		action,
		bar,
		widgets.NewParagraph("").ContentSignal(d.note.AsReadonly()).FontSize(p.Type.BodySmall.FontSize).Color(p.Text()).Font(app.Prose),
		widgets.NewParagraph("").ContentSignal(pick(func(m library.Model, _ int) string { return "From " + m.Ref().String() })).
			FontSize(p.Type.LabelSmall.FontSize).Color(p.Muted()).Font(app.Mono),
	).Gap(p.Space.S).CrossAlign(primitives.CrossAxisStretch))

	empty := widgets.Hide(none, widgets.NewParagraph("Pick a model to see how it runs here.").
		FontSize(p.Type.BodyMedium.FontSize).Color(p.Muted()).Font(app.Prose))

	return primitives.Box(primitives.VBox(empty, details).CrossAlign(primitives.CrossAxisStretch)).
		Padding(p.Space.M+p.Space.XS).
		Rounded(p.Shape.Large).
		BorderStyle(1, p.Colors.OutlineVariant).
		Background(p.Colors.SurfaceContainer)
}

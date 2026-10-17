package screen

import (
	"fmt"
	"strings"
	"sync/atomic"

	"github.com/gogpu/ui/core/button"
	"github.com/gogpu/ui/core/chip"
	"github.com/gogpu/ui/core/collapsible"
	"github.com/gogpu/ui/core/progressbar"
	"github.com/gogpu/ui/core/scrollview"
	"github.com/gogpu/ui/core/textfield"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/common/crash"
	"github.com/jitllm/jitllm/common/hardware"
	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/widgets"
)

// Devices is the hardware page and the placement control: what this machine
// offers jitllm, which models hold what of it, and where the blocks actually
// landed. They are on one page because each is read against the others.
// Everything it renders is a signal, so the probe can land after the window
// opens without rebuilding the tree.
func Devices(sh *app.Shell) widget.Widget {
	StartProbe(sh)
	p := sh.P

	body := primitives.VBox(
		devicesHeader(sh),
		machineStats(sh),
		card(sh, "Loaded models", openModels(sh)),
		// The loaded model's placement: the part of this page that moves.
		card(sh, "Where the model is", primitives.VBox(
			placement(sh),
			primitives.Box().Height(p.Space.M),
			relocate(sh),
		).CrossAlign(primitives.CrossAxisStretch)),
		primitives.HBox(
			primitives.Expanded(card(sh, "GPUs", devices(sh))),
			primitives.Expanded(card(sh, "This computer", primitives.VBox(host(sh), budgets(sh)).Gap(p.Space.S))),
		).Gap(p.Space.M),
		card(sh, "Kernels", kernels(sh)),
	).Gap(p.Space.M).CrossAlign(primitives.CrossAxisStretch).Padding(p.Space.L)

	return primitives.Box(primitives.VBox(
		primitives.Expanded(scrollview.New(body, scrollview.PainterOpt(p.Scrollbar))),
	)).Background(p.Background())
}

// card is one titled panel of the page, in Discover's card style.
func card(sh *app.Shell, title string, content widget.Widget) widget.Widget {
	p := sh.P
	return primitives.Box(primitives.VBox(
		p.Role(primitives.Text(title).Color(p.Text()), p.Type.TitleSmall).Bold(),
		content,
	).Gap(p.Space.S+p.Space.XS).CrossAlign(primitives.CrossAxisStretch)).
		Padding(p.Space.M+p.Space.XS).
		Rounded(p.Shape.Large).
		BorderStyle(1, p.Colors.OutlineVariant).
		Background(p.Colors.SurfaceContainer)
}

// machineStats is the machine in four numbers: the processor, the memory,
// the bandwidth a token is read at -- which decides decode speed -- and the GPU.
func machineStats(sh *app.Shell) widget.Widget {
	p := sh.P
	st := sh.Store
	gpu := func(r hardware.Report) (hardware.GPU, int, bool) {
		n, first, ok := 0, hardware.GPU{}, false
		for _, g := range r.GPUs {
			if g.Unified {
				continue
			}
			if !ok {
				first, ok = g, true
			}
			n++
		}
		return first, n, ok
	}
	probed := func(f func(hardware.Report) string) state.ReadonlySignal[string] {
		return machine(sh, func(r hardware.Report) string {
			if !r.Probed {
				return "reading\u2026"
			}
			return f(r)
		})
	}
	return primitives.HBox(
		primitives.Expanded(stat(sh, widgets.IconMachine, "Processor",
			probed(func(r hardware.Report) string { return hardware.ShortCPU(r.CPU) }),
			probed(func(r hardware.Report) string {
				if len(r.ECores) > 0 {
					return fmt.Sprintf("%d P + %d E cores", len(r.PCores), len(r.ECores))
				}
				return fmt.Sprintf("%d cores", len(r.PCores))
			}))),
		primitives.Expanded(stat(sh, widgets.IconMemory, "Memory",
			probed(func(r hardware.Report) string { return app.Bytes(r.RAMTotal) }),
			probed(func(r hardware.Report) string { return app.Bytes(r.MemBudget) + " for weights" }))),
		primitives.Expanded(stat(sh, widgets.IconGauge, "Bandwidth",
			state.NewComputed(func() string {
				if w := st.MemWall.Get(); w > 0 {
					return app.GBs(w)
				}
				return "--"
			}, st.MemWall.AsReadonly()),
			// The last reply against the wall, when there was one: how close
			// decode came to what the memory can give.
			state.NewComputed(func() string {
				g, w := st.GBs.Get(), st.MemWall.Get()
				if g > 0 && w > 0 {
					return fmt.Sprintf("last reply used %.0f%%", 100*g/w)
				}
				return "sets how fast replies are"
			}, st.GBs.AsReadonly(), st.MemWall.AsReadonly()))),
		primitives.Expanded(stat(sh, widgets.IconGPU, "GPU",
			probed(func(r hardware.Report) string {
				if g, _, ok := gpu(r); ok {
					return hardware.ShortGPU(g.Name)
				}
				return "None"
			}),
			probed(func(r hardware.Report) string {
				g, n, ok := gpu(r)
				if !ok {
					return "replies run on the CPU"
				}
				s := app.Bytes(g.Total) + " " + strings.ToUpper(g.API)
				if n > 1 {
					s += fmt.Sprintf(", %d more", n-1)
				}
				return s
			}))),
	).Gap(p.Space.M)
}

// stat is one of the four: an icon, what it is, the number, and what the
// number means here.
func stat(sh *app.Shell, icon widgets.Icon, label string, value, note state.ReadonlySignal[string]) widget.Widget {
	p := sh.P
	one := func(sig state.ReadonlySignal[string], size float32, c widget.Color) *widgets.Paragraph {
		return widgets.NewParagraph("").ContentSignal(sig).FontSize(size).Color(c).MaxLines(1).Font(app.Prose)
	}
	return primitives.Box(primitives.VBox(
		primitives.HBox(
			widgets.NewIconTile(icon, 28, p.Colors.SurfaceContainerHigh, p.Colors.Primary),
			primitives.Box(p.Role(primitives.Text(label).Color(p.Muted()), p.Type.LabelLarge)).PaddingTop(6),
		).Gap(p.Space.S),
		one(value, p.Type.TitleLarge.FontSize, p.Text()).Bold(),
		one(note, p.Type.BodySmall.FontSize, p.Muted()),
	).Gap(p.Space.XS).CrossAlign(primitives.CrossAxisStretch)).
		Padding(p.Space.M).
		Rounded(p.Shape.Large).
		BorderStyle(1, p.Colors.OutlineVariant).
		Background(p.Colors.SurfaceContainer)
}

// unloadModel closes path in the attached engine.
func unloadModel(sh *app.Shell, path string) {
	eng := depsOf(sh).Engine
	if eng == nil || path == "" {
		sh.Store.Status.Set("no engine is attached: see screen.Attach")
		return
	}
	eng.Unload(path)
}

// openModels is every model held open, what host memory each holds, and the
// policy that divides it. Several open models share one budget; one runs.
func openModels(sh *app.Shell) widget.Widget {
	p := sh.P
	st := sh.Store
	palette := turnPalette(sh)
	count := state.NewComputed(func() int { return len(st.Models.Get()) }, st.Models.AsReadonly())
	list := widgets.NewColumn(count, p.Space.S, func(i int) widget.Widget {
		// Rebuilt when the count moves; every field reads the row's model again,
		// since one closing and another opening keeps the count.
		at := func() app.LoadedModel {
			if ms := st.Models.Get(); i < len(ms) {
				return ms[i]
			}
			return app.LoadedModel{}
		}
		held := reactive(sh, st.Models.AsReadonly(), func() string {
			s := app.Bytes(at().Grant) + " of host memory"
			if at().Pinned {
				s += ", capped"
			}
			return s
		})
		active := state.NewComputed(func() bool { return at().Active }, st.Models.AsReadonly())
		return primitives.HBox(
			widgets.NewLetterTile("", 32, p.Colors.Primary, p.Colors.OnPrimary).Bind(state.NewComputed(func() widgets.TileLook {
				m := at()
				return widgets.TileLook{Letter: strings.ToUpper(firstRune(m.Name)), Bg: palette[m.Colour%len(palette)]}
			}, st.Models.AsReadonly())),
			primitives.Expanded(primitives.VBox(
				widgets.NewParagraph("").ContentSignal(reactive(sh, st.Models.AsReadonly(), func() string { return modelLabel(at().Path) })).
					FontSize(p.Type.BodyMedium.FontSize).Color(p.Text()).MaxLines(1).Font(app.Prose).Bold(),
				line(sh, reactive(sh, st.Models.AsReadonly(), func() string {
					m := at()
					return fmt.Sprintf("%s, %d layers", m.Arch, m.Blocks)
				})),
			).Gap(2)),
			widgets.Hide(active, primitives.Box(widgets.NewStatusPill(state.NewComputed(func() widgets.Status {
				return widgets.Status{Label: "In chat", Dot: p.Colors.Primary}
			}, active), p.Muted(), p.Type.BodySmall.FontSize)).PaddingTop(4)),
			primitives.Box(primitives.Text("").ContentSignal(held).FontSize(p.Type.BodySmall.FontSize).Color(p.Text())).PaddingTop(4),
			// Unloading is the only way to give a model's memory back.
			widgets.NewIconButton(widgets.IconTrash, "Unload", func() { unloadModel(sh, at().Path) }, iconColors(sh)),
		).Gap(p.Space.S + p.Space.XS)
	})
	rows := []widget.Widget{list}

	loads := reactive(sh, st.Loads.AsReadonly(), func() string {
		ls := st.Loads.Get()
		if len(ls) == 0 {
			return ""
		}
		s := fmt.Sprintf("Opening %s: %s", modelLabel(ls[0].Path), strings.ToLower(ls[0].Stage))
		if len(ls) > 1 {
			s += fmt.Sprintf(", %d more waiting", len(ls)-1)
		}
		return s
	})
	none := reactive(sh, st.Models.AsReadonly(), func() string {
		if len(st.Models.Get()) == 0 {
			return "No model is open. Load one from Models; several can be open at once."
		}
		return ""
	})

	// The policy that divides memory between open models is the user's
	// choice. Off, it is divided in load order and opening a second model does
	// not slow the first. On, the model in chat gets the rest and a switch moves
	// the bytes, paid in page-ins.
	prio := chip.New(
		chip.LabelFn(func() string { return "Memory follows the model in chat" }),
		chip.SelectedReadonlySignal(st.Priority.AsReadonly()),
		chip.OnClick(func() {
			on := !st.Priority.Get()
			st.Priority.Set(on)
			if eng := depsOf(sh).Engine; eng != nil {
				eng.SetPriority(on)
			}
		}),
		chip.PainterOpt(p.Chip),
	)
	more := button.New(
		button.TextOpt("Load another"),
		button.SizeOpt(button.Small),
		button.VariantOpt(button.Outlined),
		button.OnClick(sh.GoModels),
		button.PainterOpt(p.Button),
	)

	rows = append(rows,
		wrapped(sh, none),
		wrapped(sh, loads),
		primitives.HBox(prio, primitives.Expanded(primitives.Box()), more).Gap(p.Space.S),
		note(sh,
			"Loaded models share the host memory, and each stays usable. With the toggle off",
			"each keeps what it was given in load order. On, the model in chat takes the rest.",
		),
	)
	return primitives.VBox(rows...).Gap(p.Space.S).CrossAlign(primitives.CrossAxisStretch)
}

// firstRune is s's first letter, or "".
func firstRune(s string) string {
	for _, r := range s {
		return string(r)
	}
	return ""
}

// --- the probe -------------------------------------------------------------

// probing guards the probe against a second caller. It is package state
// because a theme swap rebuilds the screen, and two concurrent backend.Open
// calls is a known process-wide failure.
var probing atomic.Bool

// StartProbe runs the hardware probe once, on a worker, if it has not run.
// It is exported so an integrator can start it at startup.
func StartProbe(sh *app.Shell) {
	if sh == nil || sh.Store == nil || sh.Store.Machine.Get().Probed {
		return
	}
	Reprobe(sh)
}

// Reprobe re-reads the machine even if it has been read before. It refuses
// while a model is loaded: backend.Open opens and closes every backend, which
// contends with a live tier.GPU. The button is disabled too, but a caller is
// not a button.
func Reprobe(sh *app.Shell) {
	if sh == nil || sh.Store == nil || sh.Store.Loaded.Get() {
		return
	}
	if !probing.CompareAndSwap(false, true) {
		return
	}
	sh.SetStatus("reading the machine...")
	// Posted rather than spawned here: Devices runs inside Shell.Build, before
	// the window loop, and the first backend.Open must not race the toolkit's
	// own device creation.
	sh.Post(func() {
		go func() {
			defer crash.Recover("hardware probe")
			defer probing.Store(false)
			r := depsOf(sh).Machine.Probe()
			// Published on the UI goroutine, not here. A Set is safe from any
			// goroutine, but one that lands while a frame is laying out loses
			// its invalidation -- the widget's cache is not valid yet, so the
			// mark is dropped as already pending -- and the frame then caches
			// the old size for good. The stage caught Discover's figures blank.
			sh.Post(func() {
				sh.Store.Machine.Set(r.Machine())
				sh.Store.MemWall.Set(r.MemWall)
				sh.SetStatus(hardware.ProbeSummary(r))
			})
		}()
	})
}

// --- chrome ----------------------------------------------------------------

func devicesHeader(sh *app.Shell) widget.Widget {
	p := sh.P
	reprobe := button.New(
		button.TextOpt("Re-probe"),
		button.SizeOpt(button.Small),
		button.DisabledReadonlySignal(sh.Store.Loaded.AsReadonly()),
		button.OnClick(func() { Reprobe(sh) }),
		button.PainterOpt(p.Button),
	)
	copyBtn := button.New(
		button.TextOpt("Copy report"),
		button.SizeOpt(button.Small),
		button.VariantOpt(button.Outlined),
		button.OnClick(func() {
			r, _ := depsOf(sh).Machine.Cached()
			sh.Copy(ReportText(r))
			sh.SetStatus("hardware report copied")
		}),
		button.PainterOpt(p.Button),
	)

	// The reason the button is off, and it only appears when it is off. The
	// cause stays here rather than on screen: a probe opens every backend
	// again, which contends with the live one (see Reprobe).
	why := reactive(sh, sh.Store.Loaded.AsReadonly(), func() string {
		if sh.Store.Loaded.Get() {
			return "Re-probe is off while a model is loaded. Unload the models to probe again."
		}
		return ""
	})

	return primitives.VBox(
		primitives.HBox(
			primitives.Expanded(primitives.VBox(
				p.Role(primitives.Text("Machine").Color(p.Text()), p.Type.HeadlineSmall).Bold(),
				widgets.NewParagraph("What this computer gives jitllm, and where the loaded models sit in it.").
					FontSize(p.Type.BodyMedium.FontSize).Color(p.Muted()).Font(app.Prose),
			).Gap(p.Space.XS).CrossAlign(primitives.CrossAxisStretch)),
			reprobe,
			copyBtn,
		).Gap(p.Space.S),
		line(sh, why),
	).Gap(p.Space.XS).CrossAlign(primitives.CrossAxisStretch)
}

// section is one titled, collapsible block, for the sampling panel.
func section(sh *app.Shell, title string, content widget.Widget) widget.Widget {
	// A card, so the sections have visible boundaries.
	return sh.P.Card(collapsible.New(
		collapsible.Title(title),
		collapsible.Content(primitives.Box(content).PaddingXY(16, 12)),
		collapsible.Expanded(true),
		collapsible.PainterOpt(sh.P.Collapsible),
	))
}

// --- sections --------------------------------------------------------------

func host(sh *app.Shell) widget.Widget {
	return primitives.VBox(
		kv(sh, "host", machine(sh, func(r hardware.Report) string {
			if !r.Probed {
				return "reading..."
			}
			return fmt.Sprintf("%s/%s   %s", r.GOOS, r.GOARCH, r.GoVersion)
		})),
		kv(sh, "cpu", machine(sh, func(r hardware.Report) string { return r.CPU })),
		kv(sh, "cores", machine(sh, hardware.CoreLine)),
		kv(sh, "decode pool", machine(sh, hardware.DecodeLine)),
		kv(sh, "memory", machine(sh, func(r hardware.Report) string {
			if r.RAMTotal == 0 {
				return "unknown"
			}
			return app.Bytes(r.RAMTotal) + " total"
		})),
		// The pool is what the engine builds, not the topology: E-cores and
		// hyperthreads do not help decode (AGENTS.md RULE 5).
		note(sh,
			"The decode pool is the cores that write the reply. It uses P-cores only:",
			"adding E-cores or hyperthreads makes generation slower, not faster.",
		),
	)
}

func budgets(sh *app.Shell) widget.Widget {
	return primitives.VBox(
		kv(sh, "weight budget", machine(sh, func(r hardware.Report) string {
			if r.MemBudget == 0 {
				return "not limited here"
			}
			return app.Bytes(r.MemBudget)
		})),
		kv(sh, "cgroup limit", machine(sh, func(r hardware.Report) string {
			if r.MemLimit == 0 {
				return "none"
			}
			return app.Bytes(r.MemLimit)
		})),
		kv(sh, "collector cap", machine(sh, func(r hardware.Report) string {
			if r.GCBudgetCap == 0 {
				return "not limited here"
			}
			return fmt.Sprintf("%s of %s usable heap", app.Bytes(r.GCBudgetCap), app.Bytes(r.GCUsable))
		})),
		// Page frames are permanently live, so a weight budget near the memory limit
		// leaves the Go collector a goal it cannot reach (AGENTS.md RULE 2f).
		note(sh,
			"The weight budget is how much memory model weights may use. Keep a budget",
			"you set below the collector cap: above it the engine spends much of its",
			"time freeing memory and runs slower.",
		),
	)
}

func kernels(sh *app.Shell) widget.Widget {
	rows := []widget.Widget{
		primitives.HBox(
			col(sh, sh.P.Role(primitives.Text("format").Color(sh.P.Muted()), sh.P.Type.LabelMedium), 90),
			col(sh, sh.P.Role(primitives.Text("row-major (GGUF)").Color(sh.P.Muted()), sh.P.Type.LabelMedium), 170),
			sh.P.Role(primitives.Text("packed  <- used to run models").Color(sh.P.Text()), sh.P.Type.LabelMedium),
		),
	}
	for i := range hardware.FormatNames() {
		rows = append(rows, primitives.HBox(
			col(sh, line(sh, machine(sh, func(r hardware.Report) string { return hardware.KernelCell(r, i, 0) })), 90),
			col(sh, line(sh, machine(sh, func(r hardware.Report) string { return hardware.KernelCell(r, i, 1) })), 170),
			line(sh, machine(sh, func(r hardware.Report) string { return hardware.KernelCell(r, i, 2) })),
		))
	}

	rows = append(rows,
		primitives.Box().Height(6),
		kv(sh, "isa", machine(sh, func(r hardware.Report) string { return r.ISA })),
		kv(sh, "int8 dot", machine(sh, func(r hardware.Report) string { return r.DotKind })),
		// Two columns because the row-major family reads GGUF blocks, which a
		// container never decodes through; a host can read none there and still be
		// fully generated through the packed one.
		note(sh,
			"Models run through the packed column. The row-major kernels only read GGUF",
			"files, which are converted before loading, so `none` there is normal.",
		),
	)
	return primitives.VBox(rows...)
}

// deviceSlots is how many devices this page can draw. An unused slot has
// empty text, measures zero and draws nothing, so a fixed tree renders a
// variable list.
const deviceSlots = 8

func devices(sh *app.Shell) widget.Widget {
	p := sh.P
	rows := []widget.Widget{
		wrapped(sh, machine(sh, func(r hardware.Report) string {
			switch {
			case !r.Probed:
				return "reading\u2026"
			case len(r.GPUs) == 0:
				return "No CUDA, Vulkan or Metal device opened here. This is a CPU machine, not an error."
			}
			return ""
		})),
	}
	for i := 0; i < deviceSlots; i++ {
		at := func(r hardware.Report) (hardware.GPU, bool) {
			if i >= len(r.GPUs) {
				return hardware.GPU{}, false
			}
			return r.GPUs[i], true
		}
		show := state.NewComputed(func() bool {
			r, _ := depsOf(sh).Machine.Cached()
			_, ok := at(r)
			return ok
		}, sh.Store.Machine.AsReadonly())
		gpu := func(f func(hardware.GPU) string) state.ReadonlySignal[string] {
			return machine(sh, func(r hardware.Report) string {
				if g, ok := at(r); ok {
					return f(g)
				}
				return ""
			})
		}
		// The card's memory in use, as a bar: what is left is what the next
		// load can place there.
		used := state.NewComputed(func() float64 {
			r, _ := depsOf(sh).Machine.Cached()
			if g, ok := at(r); ok && g.Total > 0 {
				return float64(g.Total-min(g.Free, g.Total)) / float64(g.Total)
			}
			return 0
		}, sh.Store.Machine.AsReadonly())
		known := state.NewComputed(func() bool {
			r, _ := depsOf(sh).Machine.Cached()
			g, ok := at(r)
			return ok && g.Total > 0
		}, sh.Store.Machine.AsReadonly())
		rows = append(rows, widgets.Hide(show, primitives.VBox(
			widgets.NewParagraph("").ContentSignal(gpu(func(g hardware.GPU) string { return g.Name })).
				FontSize(p.Type.BodyMedium.FontSize).Color(p.Text()).MaxLines(1).Font(app.Prose).Bold(),
			line(sh, gpu(DeviceStat)),
			// Expanded in a row, because a progress bar laid out loose is its
			// own preferred 200 px whatever the column is.
			widgets.Hide(known, primitives.HBox(primitives.Expanded(progressbar.New(
				progressbar.ValueReadonlySignal(used),
				progressbar.ColorSchemeOpt(p.ProgressBar),
				progressbar.Height(6),
			)))),
			wrapped(sh, gpu(DeviceNote)),
		).Gap(p.Space.XS).CrossAlign(primitives.CrossAxisStretch).PaddingTop(p.Space.M)))
	}

	for i := 0; i < deviceSlots; i++ {
		rows = append(rows, wrapped(sh, machine(sh, func(r hardware.Report) string {
			if len(r.Vulkan) < 2 || i >= len(r.Vulkan) {
				return ""
			}
			return VulkanLine(r.Vulkan[i])
		})))
	}

	// A shared-memory device is not a second pool: its bytes are subtracted
	// from the host budget, never added.
	rows = append(rows, primitives.Box(note(sh,
		"A GPU that shares memory with the system, such as an integrated GPU or Apple",
		"Silicon, adds no memory: what it uses comes out of the host budget.",
	)).PaddingTop(p.Space.M))
	// No gap: a hidden slot would still take one, and eight empty slots read
	// as a hole. Each row carries its own space above it.
	return primitives.VBox(rows...).CrossAlign(primitives.CrossAxisStretch)
}

// wrapped is a muted paragraph that wraps, and takes no room when empty --
// not even a gap, since it carries its own space above it.
func wrapped(sh *app.Shell, sig state.ReadonlySignal[string]) widget.Widget {
	show := state.NewComputed(func() bool { return sig.Get() != "" }, sig)
	return widgets.Hide(show, primitives.Box(widgets.NewParagraph("").
		ContentSignal(sig).
		FontSize(sh.P.Type.BodySmall.FontSize).
		LineHeight(sh.P.Type.BodySmall.LineHeight/sh.P.Type.BodySmall.FontSize).
		Color(sh.P.Muted()).
		Font(app.Prose)).PaddingTop(sh.P.Space.XS))
}

func nextLoad(sh *app.Shell) widget.Widget {
	spec := sh.Store.DeviceSpec

	// The budget lives in the persisted config: the Store has no field for it
	// and a per-build signal would forget it on a theme swap.
	budget := state.NewSignal(sh.Cfg.MaxMem)
	specNote := state.NewSignal("")
	specErr := state.NewSignal("")
	budgetNote := state.NewSignal("")
	budgetErr := state.NewSignal("")

	checkSpec := func(s string) {
		if got, err := hardware.ValidateSpec(s); err != nil {
			specNote.Set("")
			specErr.Set(err.Error())
		} else {
			specErr.Set("")
			specNote.Set(got)
		}
	}
	checkBudget := func(s string) {
		r, _ := depsOf(sh).Machine.Cached()
		if got, err := hardware.ValidateBudget(s, r.MemBudget, r.GCBudgetCap); err != nil {
			budgetNote.Set("")
			budgetErr.Set(err.Error())
		} else {
			budgetErr.Set("")
			budgetNote.Set(got)
		}
	}
	// The block map is validated against the spec it names devices from, so a
	// change to either re-checks it.
	place := sh.Store.Placement
	placeErr := state.NewSignal("")
	checkPlace := func(string) {
		if err := hardware.ValidatePlacement(place.Get(), spec.Get()); err != nil {
			placeErr.Set(err.Error())
		} else {
			placeErr.Set("")
		}
	}
	checkSpec(spec.Get())
	checkBudget(budget.Get())
	checkPlace("")

	set := func(s string) {
		spec.Set(s)
		checkSpec(s)
		checkPlace("")
	}

	presets := []string{"auto", "all", "cpu", "gpu", "gpu:0", "cuda:0", "vulkan:0", "vulkan:1", "metal"}
	chips := make([]widget.Widget, 0, len(presets))
	for _, p := range presets {
		chips = append(chips, chip.New(
			chip.Label(p),
			chip.Selectable(true),
			// A controlled chip: it reads the spec rather than remembering a
			// click, so typing a spec by hand moves the highlight too.
			chip.SelectedReadonlySignal(state.NewComputed(func() bool {
				return spec.Get() == p
			}, spec.AsReadonly())),
			chip.OnClick(func() { set(p) }),
			chip.PainterOpt(sh.P.Chip),
		))
	}

	specField := textfield.New(
		textfield.ValueSignal(spec),
		textfield.Placeholder("auto"),
		textfield.OnChange(func(s string) {
			checkSpec(s)
			checkPlace("")
		}),
		textfield.PainterOpt(sh.P.TextField),
	)
	placeField := textfield.New(
		textfield.ValueSignal(place),
		textfield.Placeholder("0-15=cuda:0,16-31=vulkan:0"),
		textfield.OnChange(checkPlace),
		textfield.PainterOpt(sh.P.TextField),
	)
	budgetField := textfield.New(
		textfield.ValueSignal(budget),
		textfield.Placeholder("the engine's default"),
		textfield.OnChange(func(s string) {
			// Cfg is what the loader reads and what the shell persists.
			sh.Cfg.MaxMem = s
			sh.SaveSoon()
			checkBudget(s)
		}),
		textfield.PainterOpt(sh.P.TextField),
	)

	return primitives.VBox(
		sh.P.Role(primitives.Text("Run models on").Color(sh.P.Muted()), sh.P.Type.LabelMedium),
		primitives.HBox(chips...).Gap(sh.P.Space.S),
		sh.P.Field(specField).Width(360),
		line(sh, specNote.AsReadonly()),
		line(sh, specErr.AsReadonly()).Color(sh.P.Colors.Error),
		primitives.Box().Height(8),
		sh.P.Role(primitives.Text("Block placement (optional)").Color(sh.P.Muted()), sh.P.Type.LabelMedium),
		sh.P.Field(placeField).Width(360),
		line(sh, placeErr.AsReadonly()).Color(sh.P.Colors.Error),
		note(sh,
			"Blocks to devices: 0-15=cuda:0,16-31=vulkan:0, head=cuda:0, host for the CPU.",
			"Name each device in Run models on. Empty lets jitllm place the blocks.",
		),
		primitives.Box().Height(8),
		sh.P.Role(primitives.Text("Memory for model weights").Color(sh.P.Muted()), sh.P.Type.LabelMedium),
		sh.P.Field(budgetField).Width(360),
		line(sh, budgetNote.AsReadonly()),
		line(sh, budgetErr.AsReadonly()).Color(sh.P.Colors.Error),
		// A spec that names a device it cannot open is an error, not a fall back to
		// the host. A budget one page short of fitting is a cliff, not a gradient.
		note(sh,
			"Both apply to the next load; an open model keeps its placement. Naming a",
			"device this machine cannot open is an error, not a quiet switch to the CPU.",
			"A budget one layer short of the model runs much slower than one that fits.",
		),
	)
}

func placement(sh *app.Shell) widget.Widget {
	blocks := reactive(sh, sh.Store.BlockMap.AsReadonly(), func() string {
		return PlacementLine(sh.Store.BlockMap.Get())
	})
	pager := reactive(sh, sh.Store.Pager.AsReadonly(), func() string {
		return PagerLine(sh.Store.Pager.Get())
	})
	declines := reactive(sh, sh.Store.DeclineReport.AsReadonly(), func() string {
		if s := sh.Store.DeclineReport.Get(); s != "" {
			return "declined: " + s
		}
		return ""
	})

	return primitives.VBox(
		line(sh, blocks).Color(sh.P.Text()),
		primitives.Box().Height(8),
		memoryMap(sh),
		primitives.Box().Height(6),
		kv(sh, "device", sh.Store.DeviceReport.AsReadonly()),
		line(sh, declines).Color(sh.P.Colors.Error),
		// Requests beside bytes: the pager's cost is per request.
		kv(sh, "pager", pager),
		// Bytes against a capacity, never summed across lanes: a pool is per
		// heap, so one card reached through two APIs, or a unified-memory
		// device and the host, can be the same bytes.
		note(sh,
			"Each bar is one kind of memory against what the model may use of it. Two bars can be",
			"the same memory (one card through two APIs, or a GPU sharing system memory), so don't add them.",
		),
		note(sh,
			"If the pager shows pages going out, the model does not fit in the weight",
			"budget and blocks are read from disk again as it runs, which is slow. Raise",
			"the budget if the machine has the memory.",
		),
	)
}

// --- small builders --------------------------------------------------------

// machine derives a string from the last probe. It depends on the published
// app.MachineReport and reads the richer hardware.Report cache; both are
// filled by the same probe call, so the pair cannot be torn.
func machine(sh *app.Shell, f func(hardware.Report) string) state.ReadonlySignal[string] {
	return state.NewComputed(func() string {
		r, _ := depsOf(sh).Machine.Cached()
		return f(r)
	}, sh.Store.Machine.AsReadonly())
}

// reactive derives a string from any one signal.
func reactive[T any](_ *app.Shell, dep state.ReadonlySignal[T], f func() string) state.ReadonlySignal[string] {
	return state.NewComputed(f, dep)
}

// line is one row of text. MaxLines(1) matters: the measurer estimates a
// wrapped height while the canvas draws one line, so an uncapped long string
// reserves three lines and paints one. Prose that must wrap uses [note].
func line(sh *app.Shell, sig state.ReadonlySignal[string]) *widgets.Paragraph {
	// A capped Paragraph, not Text with Ellipsis(): Ellipsis affects only
	// measuring, and Canvas.DrawText does not clip, so the text overdrew its
	// neighbours.
	return widgets.NewParagraph("").
		ContentSignal(sig).
		FontSize(sh.P.Type.BodySmall.FontSize).
		LineHeight(sh.P.Type.BodySmall.LineHeight / sh.P.Type.BodySmall.FontSize).
		Color(sh.P.Muted()).
		MaxLines(1).
		Font(app.Prose)
}

func indent(sh *app.Shell, sig state.ReadonlySignal[string]) widget.Widget {
	return primitives.HBox(
		primitives.Box().Width(16),
		primitives.Expanded(line(sh, sig)),
	)
}

func kv(sh *app.Shell, label string, sig state.ReadonlySignal[string]) widget.Widget {
	return widgets.KVSignal(label, sig, sh.P.Muted(), sh.P.Text(), sh.P.Type.BodySmall.FontSize)
}

func col(_ *app.Shell, w widget.Widget, width float32) widget.Widget {
	return primitives.Box(w).Width(width)
}

// note is the explanatory prose under a control: one wrapping paragraph,
// indented and muted so it reads as an aside. Callers pass short lines for
// readability in source; they are joined and rewrapped to the given width.
// It is a plain-words hint for the user; measurements and engine reasons go
// in a Go comment beside the call (TestMachineHelpIsPlainAndShort holds the
// Machine page to that).
func note(sh *app.Shell, lines ...string) widget.Widget {
	body := widgets.NewParagraph(strings.Join(lines, " ")).
		FontSize(sh.P.Type.BodySmall.FontSize).
		LineHeight(sh.P.Type.BodySmall.LineHeight / sh.P.Type.BodySmall.FontSize).
		Color(sh.P.Muted()).
		Font(app.Prose)

	return primitives.Box(body).PaddingXY(sh.P.Space.S, sh.P.Space.XS)
}

// --- pure formatting -------------------------------------------------------

// DeviceStat is a device's API, slots and memory; see [hardware.DeviceStat].
func DeviceStat(g hardware.GPU) string { return hardware.DeviceStat(g) }

// DeviceLine is one opened device; see [hardware.DeviceLine].
func DeviceLine(g hardware.GPU) string { return hardware.DeviceLine(g) }

// DeviceNote is what else is true of a device; see [hardware.DeviceNote].
func DeviceNote(g hardware.GPU) string { return hardware.DeviceNote(g) }

// VulkanLine is one Vulkan device; see [hardware.VulkanLine].
func VulkanLine(v hardware.VulkanDev) string { return hardware.VulkanLine(v) }

// PlacementLine counts where the blocks went; see [session.PlacementLine].
func PlacementLine(b []byte) string { return session.PlacementLine(b) }

// PagerLine is the container pager's counters; see [session.PagerLine].
func PagerLine(p app.PagerStat) string { return session.PagerLine(p) }

// ReportText is the whole probe as plain text; see [hardware.ReportText].
func ReportText(r hardware.Report) string { return hardware.ReportText(r) }

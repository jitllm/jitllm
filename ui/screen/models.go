package screen

import (
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/gogpu/gogpu"
	"github.com/gogpu/ui/core/button"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/common/catalog"
	"github.com/jitllm/jitllm/common/discover"
	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/widgets"
)

// The Models screen: the containers on disk, what each says about itself, and
// the action that matters -- load one. The files to convert are the Convert
// tab's (convert.go); the one scan here fills both lists.
//
// catalog.Probe goes through jlm.Open, which reads the header, config,
// vocabulary and tensor table but no pages, so listing is cheap whatever the
// models weigh. Staleness is read structurally: catalog.ReadVersion compares
// the file's version word against this build's rather than matching jlm.Open's
// error text.

// Models builds the catalog screen. It is an [app.ScreenFunc].
func Models(sh *app.Shell) widget.Widget {
	m := modelsFor(sh)
	p := sh.P
	add := button.New(button.TextOpt("Add folder"), button.SizeOpt(button.Small),
		button.VariantOpt(button.Outlined), button.OnClick(m.addFolder), button.PainterOpt(p.Button))
	rescan := button.New(button.TextOpt("Rescan"), button.SizeOpt(button.Small),
		button.VariantOpt(button.Outlined), button.OnClick(m.rescan), button.PainterOpt(p.Button))
	head := pageHead(sh, "Models",
		"The models in your folders, ready to run. Pick one to see what it is; Load opens it for chat.",
		add, rescan)
	return listPage(sh, head, m.list(), m.detailSide(), discoverPanelWidth)
}

// --- the screen's own state -------------------------------------------------

// modelsScreen is the state this screen needs and the shared Store has no
// field for: the engine hooks, the scan, the sort and the derived strings the
// detail pane binds to. It is keyed by shell rather than held in a package
// variable: a screen is rebuilt on theme change, and two shells (a test and a
// window) must not share one convert form.
type modelsScreen struct {
	sh    *app.Shell
	store *app.Store

	// Derived, so the detail pane repaints when the selection moves.
	dPath, dStatus, dLayers, dGeom, dMixture state.ReadonlySignal[string]
	dVision, dChat, dPages, dChunk, dStream  state.ReadonlySignal[string]
	dDense, dWriter, dQuant                  state.ReadonlySignal[string]
	// dHint says in one sentence why a row cannot be loaded; empty when it can.
	dHint                           state.ReadonlySignal[string]
	loadDisabled, reconvertDisabled state.ReadonlySignal[bool]
	loadLabel                       state.ReadonlySignal[string]

	mu          sync.Mutex
	scanRunning bool
}

// modelsFor returns this shell's screen state, creating it -- and kicking off
// the first scan -- exactly once.
func modelsFor(sh *app.Shell) *modelsScreen {
	m, created := app.Local(sh, func() *modelsScreen { return newModelsScreen(sh) })
	if created {
		m.rescan()
		// Folders added or removed anywhere -- this tab, a drop, Settings --
		// change what is on disk to show. Set on the UI goroutine, so this runs
		// there too.
		sh.Store.ModelDirs.SubscribeForever(func([]string) { m.rescan() })
	}
	return m
}

func newModelsScreen(sh *app.Shell) *modelsScreen {
	m := &modelsScreen{
		sh:    sh,
		store: sh.Store,
	}

	m.dPath = m.field(func(e catalog.Entry) string { return e.Path })
	// A loaded model says so here as on its row, not "ready to load".
	m.dStatus = state.NewComputed(func() string {
		e, ok := m.selected()
		if !ok {
			return "--"
		}
		if openState(m.store, e.Path) != "" {
			return "loaded"
		}
		return entryStatus(e)
	}, m.store.Catalog.AsReadonly(), m.store.CatalogSel.AsReadonly(), m.store.Models.AsReadonly())
	m.dLayers = m.field(entryLayers)
	m.dGeom = m.field(entryGeometry)
	m.dMixture = m.field(entryMixture)
	m.dVision = m.field(entryVision)
	m.dChat = m.field(entryChat)
	m.dPages = m.field(entryPages)
	m.dChunk = m.field(entryChunk)
	m.dStream = m.field(entryStream)
	m.dDense = m.field(entryDense)
	m.dQuant = m.field(func(e catalog.Entry) string {
		if e.Quant == "" {
			return "--"
		}
		return e.Quant
	})
	m.dWriter = m.field(func(e catalog.Entry) string {
		if e.Writer == "" {
			return "--"
		}
		return e.Writer
	})

	m.dHint = state.NewComputed(func() string {
		e, ok := m.selected()
		if !ok {
			return ""
		}
		return entryHint(e)
	}, m.store.Catalog.AsReadonly(), m.store.CatalogSel.AsReadonly())

	// Several models can be loading: the engine opens them in the order asked.
	// The one already on its way is not asked for twice.
	m.loadDisabled = state.NewComputed(func() bool {
		e, ok := m.store.SelectedEntry()
		_, busy := m.store.LoadingOf(e.Path)
		return !ModelLoadable(e, ok) || busy
	}, m.store.Catalog.AsReadonly(), m.store.CatalogSel.AsReadonly(), m.store.Loads.AsReadonly())
	m.loadLabel = state.NewComputed(func() string {
		e, ok := m.store.SelectedEntry()
		if l, busy := m.store.LoadingOf(e.Path); ok && busy {
			if l.Stage == "Waiting" {
				return "Waiting\u2026"
			}
			return "Loading\u2026"
		}
		// A loaded model is chatted with, never loaded twice.
		if openState(m.store, e.Path) != "" {
			if m.store.Active.Get() == e.Path {
				return "Go to chat"
			}
			return "Chat with it"
		}
		return "Load"
	}, m.store.Catalog.AsReadonly(), m.store.CatalogSel.AsReadonly(), m.store.Loads.AsReadonly(), m.store.Models.AsReadonly())
	m.reconvertDisabled = m.flag(func(e catalog.Entry, ok bool) bool { return !ModelReconvertible(e, ok) })
	return m
}

// field derives one detail row from the selected entry.
func (m *modelsScreen) field(get func(catalog.Entry) string) state.ReadonlySignal[string] {
	return state.NewComputed(func() string {
		e, ok := m.selected()
		if !ok {
			return "--"
		}
		return get(e)
	}, m.store.Catalog.AsReadonly(), m.store.CatalogSel.AsReadonly())
}

// flag derives one enabled/disabled decision from the selected entry.
func (m *modelsScreen) flag(pred func(catalog.Entry, bool) bool) state.ReadonlySignal[bool] {
	return state.NewComputed(func() bool {
		e, ok := m.selected()
		return pred(e, ok)
	}, m.store.Catalog.AsReadonly(), m.store.CatalogSel.AsReadonly())
}

// selected reads the highlighted row. It bounds-checks rather than trusting the
// index, because [app.Store.SetCatalog] publishes the slice and the row count
// in two writes and a frame can land between them.
func (m *modelsScreen) selected() (catalog.Entry, bool) {
	return m.store.SelectedEntry()
}

func (m *modelsScreen) runner() func(func()) bool {
	if eng := depsOf(m.sh).Engine; eng != nil {
		return eng.Run
	}
	return func(job func()) bool { go job(); return true }
}

// --- scanning ---------------------------------------------------------------

// rescan walks the configured directories and probes every container it
// finds. It must be called from the UI goroutine, because it snapshots
// Cfg.ModelDirs, which is written there. It publishes a fresh copy after each
// probe so names appear at once and details fill in behind, while the worker
// keeps mutating its own slice.
func (m *modelsScreen) rescan() {
	m.mu.Lock()
	if m.scanRunning {
		m.mu.Unlock()
		return
	}
	m.scanRunning = true
	m.mu.Unlock()

	dirs := append([]string(nil), m.sh.Cfg.ModelDirs...)

	go func() {
		defer m.sh.Crashes().Recover("model scan")
		defer func() {
			m.mu.Lock()
			m.scanRunning = false
			m.mu.Unlock()
		}()

		// Published on the UI goroutine: a store write that lands mid-layout
		// loses its invalidation (see Reprobe). Each publish carries its own
		// copy, since probing keeps writing into found.
		post := func(found []catalog.Entry) {
			cp := append([]catalog.Entry(nil), found...)
			m.sh.Post(func() { m.publish(cp) })
		}
		say := func(s string) { m.sh.Post(func() { m.sh.SetStatus(s) }) }

		say("scanning " + strings.Join(dirs, ", "))
		files := depsOf(m.sh).Files
		found := files.Scan(dirs)
		post(found)

		containers := 0
		for i := range found {
			if found[i].Kind != catalog.KindContainer {
				continue
			}
			files.Probe(&found[i])
			containers++
			post(found)
		}

		stale := 0
		for i := range found {
			if found[i].Stale {
				stale++
			}
		}
		msg := fmt.Sprintf("%d model(s), %d to convert", containers, len(found)-containers)
		if stale > 0 {
			msg += fmt.Sprintf(", %d stale", stale)
		}
		say(msg + " in " + strings.Join(dirs, ", "))
	}()
}

// publish hands the table a sorted copy, keeping the selection on the same
// file rather than the same row: a scan republishes once per probed container
// and a header click reorders, so an index-only selection would drift to a
// model nobody chose. It also splits the scan: containers to this table, GGUF
// and safetensors to the Convert tab's.
func (m *modelsScreen) publish(found []catalog.Entry) {
	var out, sources []catalog.Entry
	for _, e := range found {
		if e.Kind == catalog.KindContainer {
			out = append(out, e)
		} else {
			sources = append(sources, e)
		}
	}
	SortModels(out, "name", true)
	publishSources(m.store, sources)

	sel := ""
	if e, ok := m.store.SelectedEntry(); ok {
		sel = e.Path
	}
	m.store.SetCatalog(out)
	if sel != "" {
		m.store.CatalogSel.Set(rowOf(out, sel))
	}
}

// rowOf is the index of path in the published table, or -1 when the file is no
// longer there.
func rowOf(rows []catalog.Entry, path string) int {
	for i := range rows {
		if rows[i].Path == path {
			return i
		}
	}
	return -1
}

// selectPath highlights path's row and reports whether the table has one.
func (m *modelsScreen) selectPath(path string) bool {
	i := rowOf(m.store.Catalog.Get(), path)
	if i < 0 {
		return false
	}
	m.store.CatalogSel.Set(i)
	return true
}

// --- actions ----------------------------------------------------------------

func (m *modelsScreen) addFolder() {
	start := ""
	if len(m.sh.Cfg.ModelDirs) > 0 {
		start = m.sh.Cfg.ModelDirs[0]
	}
	dir := m.sh.PickDir("Add model folder", start)
	if dir == "" {
		return
	}
	if !m.sh.AddModelDir(dir) {
		m.sh.SetStatus(dir + " is already in the catalog")
		return
	}
	// The folder list moving rescans (see Models).
}

func (m *modelsScreen) load() {
	e, ok := m.selected()
	if !ModelLoadable(e, ok) {
		return
	}
	eng := depsOf(m.sh).Engine
	if eng == nil {
		m.sh.Alert("No engine wired",
			"This build has no model loader installed. An integrator wires one with "+
				"screen.Attach(sh, screen.Deps{Engine: ...}).\n\n"+
				"The container itself reads fine: "+e.Path)
		return
	}
	// The model the chat talks to is already there: loading it again would
	// close its session and place it on the device anew, for nothing.
	if openState(m.store, e.Path) != "" && m.store.Active.Get() == e.Path {
		m.sh.GoSession()
		return
	}
	m.sh.SetStatus("loading " + e.Name)
	eng.Load(e.Path)
	// The model is used on the Session tab, so Load goes there.
	m.sh.GoSession()
}

// reconvert rebuilds the selected container from its source, on the Convert
// tab, where the form and its progress are.
func (m *modelsScreen) reconvert() {
	e, ok := m.selected()
	if !ModelReconvertible(e, ok) {
		return
	}
	OfferReconvert(m.sh, e.Path)
}

// --- the list ---------------------------------------------------------------

// modelFilters() is what the convert form's Browse offers.
func modelFilters() []gogpu.FileTypeFilter {
	return []gogpu.FileTypeFilter{
		{Name: "Model weights", Extensions: []string{"gguf", "safetensors"}},
		{Name: "jitllm container", Extensions: []string{"jlm"}},
	}
}

// list is one row per container: what it is, and what it is doing here. The
// engine's numbers (page size, chunk, stream groups) are in the panel.
func (m *modelsScreen) list() widget.Widget {
	st := m.store
	col := widgets.NewColumn(st.CatalogRows.AsReadonly(), 2, func(i int) widget.Widget {
		at := func() (catalog.Entry, bool) {
			if all := st.Catalog.Get(); i < len(all) {
				return all[i], true
			}
			return catalog.Entry{}, false
		}
		text := func(f func(catalog.Entry) string) state.ReadonlySignal[string] {
			return state.NewComputed(func() string {
				if e, ok := at(); ok {
					return f(e)
				}
				return ""
			}, st.Catalog.AsReadonly(), st.Machine.AsReadonly(), st.Models.AsReadonly(), st.Loads.AsReadonly())
		}
		look := state.NewComputed(func() widgets.TileLook {
			e, _ := at()
			return tileLook(m.sh.P, e.Name)
		}, st.Catalog.AsReadonly())
		sel := state.NewComputed(func() bool { return st.CatalogSel.Get() == i }, st.CatalogSel.AsReadonly())
		return listRow(m.sh, look,
			text(func(e catalog.Entry) string { return modelLabel(e.Path) }),
			text(entryFacts),
			text(func(e catalog.Entry) string { return m.rowState(e) }),
			text(func(e catalog.Entry) string { return discover.FitWordsOf(discover.EntryFits(e, st.Machine.Get())) }),
			sel, func() { st.CatalogSel.Set(i) })
	})
	none := state.NewComputed(func() bool { return st.CatalogRows.Get() == 0 }, st.CatalogRows.AsReadonly())
	find := button.New(button.TextOpt("Find a model"), button.VariantOpt(button.Filled),
		button.OnClick(func() { m.sh.SelectTab(app.TabDownload) }), button.PainterOpt(m.sh.P.Button))
	add := button.New(button.TextOpt("Add folder"), button.VariantOpt(button.Outlined),
		button.OnClick(m.addFolder), button.PainterOpt(m.sh.P.Button))
	empty := widgets.Hide(none, panel(m.sh,
		m.sh.P.Role(primitives.Text("No models yet").Color(m.sh.P.Text()), m.sh.P.Type.TitleMedium).Bold(),
		wrapped(m.sh, state.NewSignal("Download one from Discover, or add a folder that has .jlm files in it. "+
			"A GGUF in a folder shows up on Convert, ready to turn into one.").AsReadonly()),
		primitives.HBox(find, add).Gap(m.sh.P.Space.S),
	))
	return primitives.VBox(empty, col).CrossAlign(primitives.CrossAxisStretch)
}

// rowState is the right-hand word on a row: what the model is doing in this
// app, or what stands between it and loading.
func (m *modelsScreen) rowState(e catalog.Entry) string {
	if _, ok := m.store.LoadingOf(e.Path); ok {
		return "Loading\u2026"
	}
	if s := openState(m.store, e.Path); s != "" {
		return s
	}
	switch s := entryShortStatus(e); s {
	case "ready":
		return "Ready"
	case "reconvert it":
		return "Needs reconverting"
	default:
		return strings.ToUpper(s[:1]) + s[1:]
	}
}

// openState is "Loaded" for every loaded model, whichever the chat talks to,
// and "" for one that is not loaded (session.LoadState).
func openState(st *app.Store, path string) string {
	return session.LoadState(st.Models.Get(), path)
}

// entryFacts is a row's line of facts: what someone choosing a model asks.
func entryFacts(e catalog.Entry) string {
	if e.Kind == catalog.KindContainer && !e.Probed {
		return "reading\u2026"
	}
	var parts []string
	if e.NLayer > 0 {
		parts = append(parts, fmt.Sprintf("%d layers", e.NLayer))
	}
	if e.Quant != "" {
		parts = append(parts, e.Quant)
	}
	parts = append(parts, app.Bytes(uint64(modelsNonNeg(e.Size, 0))))
	if e.HasVision {
		parts = append(parts, "sees images")
	}
	return strings.Join(parts, " \u00b7 ")
}

// --- the detail pane --------------------------------------------------------

func (m *modelsScreen) detailSide() widget.Widget {
	p := m.sh.P
	st := m.store
	some := state.NewComputed(func() bool { _, ok := m.selected(); return ok }, st.Catalog.AsReadonly(), st.CatalogSel.AsReadonly())
	none := state.NewComputed(func() bool { return !some.Get() }, some)
	name := m.field(func(e catalog.Entry) string { return modelLabel(e.Path) })
	size := m.field(func(e catalog.Entry) string { return app.Bytes(uint64(modelsNonNeg(e.Size, 0))) })
	// A label over its value, so a long value wraps instead of being cut.
	kv := func(label string, sig state.ReadonlySignal[string]) widget.Widget {
		return primitives.HBox(
			primitives.Box(p.Role(primitives.Text(label).Color(p.Muted()), p.Type.BodySmall)).Width(104),
			primitives.Expanded(widgets.NewParagraph("").ContentSignal(sig).FontSize(p.Type.BodySmall.FontSize).
				Color(p.Text()).Font(p.Prose())),
		).Gap(p.Space.S)
	}

	details := widgets.Hide(some, primitives.VBox(
		p.Role(primitives.Text("").ContentSignal(name).Color(p.Text()), p.Type.TitleLarge).Bold(),
		primitives.Box().Height(p.Space.XS),
		pair(m.sh, metric(m.sh, "Size", size), metric(m.sh, "Quant", m.dQuant)),
		pair(m.sh, metric(m.sh, "Layers", m.dLayers), metric(m.sh, "Runs", state.NewComputed(func() string {
			e, ok := m.selected()
			if !ok {
				return "--"
			}
			if w := discover.FitWordsOf(discover.EntryFits(e, st.Machine.Get())); w != "" {
				return w
			}
			return "--"
		}, st.Catalog.AsReadonly(), st.CatalogSel.AsReadonly(), st.Machine.AsReadonly()))),
		pair(m.sh, metric(m.sh, "Chat", m.field(func(e catalog.Entry) string {
			if e.HasChat {
				return "Chat and completion"
			}
			if entryRead(e) {
				return "Completion only"
			}
			return "--"
		})), metric(m.sh, "Images", m.field(func(e catalog.Entry) string {
			if e.HasVision {
				return "Sees images"
			}
			if entryRead(e) {
				return "Text only"
			}
			return "--"
		}))),
		// Said only when it matters: a model that does not fit still runs.
		wrapped(m.sh, state.NewComputed(func() string {
			e, ok := m.selected()
			mr := st.Machine.Get()
			if !ok || discover.EntryFits(e, mr) != "paged" {
				return ""
			}
			return "Larger than the " + app.Bytes(mr.MemBudget) + " memory budget, so it runs reading weights from disk, which is slow."
		}, st.Catalog.AsReadonly(), st.CatalogSel.AsReadonly(), st.Machine.AsReadonly())),
	).Gap(p.Space.S).CrossAlign(primitives.CrossAxisStretch))
	empty := widgets.Hide(none, widgets.NewParagraph("Pick a model to see what it is and load it.").
		FontSize(p.Type.BodyMedium.FontSize).Color(p.Muted()).Font(p.Prose()))

	load := button.New(
		button.TextReadonlySignal(m.loadLabel),
		button.SizeOpt(button.Large),
		button.VariantOpt(button.Filled),
		button.OnClick(m.load),
		button.DisabledReadonlySignal(m.loadDisabled),
		button.PainterOpt(p.Button),
	)
	reconvert := button.New(
		button.TextOpt("Reconvert"),
		button.SizeOpt(button.Large),
		button.VariantOpt(button.Outlined),
		button.OnClick(m.reconvert),
		button.DisabledReadonlySignal(m.reconvertDisabled),
		button.PainterOpt(p.Button),
	)
	// The engine's view of the file, for whoever is placing it by hand.
	engine := widgets.Hide(some, primitives.VBox(
		p.Role(primitives.Text("In the engine").Color(p.Muted()), p.Type.LabelMedium),
		kv("Status", m.dStatus),
		kv("Geometry", m.dGeom),
		kv("Mixture", m.dMixture),
		kv("Vision tower", m.dVision),
		kv("Pages", m.dPages),
		kv("Chunk", m.dChunk),
		kv("Stream groups", m.dStream),
		kv("Dense region", m.dDense),
		kv("Written by", m.dWriter),
		widgets.NewParagraph("").ContentSignal(m.dPath).FontSize(p.Type.LabelSmall.FontSize).
			Color(p.Muted()).Font(p.Mono()),
	).Gap(2).CrossAlign(primitives.CrossAxisStretch))

	return panel(m.sh,
		empty,
		details,
		hintBox(p, m.dHint),
		primitives.HBox(load, reconvert).Gap(p.Space.S),
		primitives.Box().Height(p.Space.XS),
		engine,
	)
}

func modelsHeading(p app.Painters, s string) widget.Widget {
	return p.Role(primitives.Text(s).Color(p.Text()), p.Type.TitleMedium)
}

func modelsDivider(p app.Painters) widget.Widget {
	return primitives.Box().Height(1).Background(p.Colors.OutlineVariant)
}

// --- pure helpers, which is where the logic worth testing lives -------------

// ModelLoadable reports whether the selected entry can be opened as a model.
// A row not yet probed is not loadable (its zero Version reads like a healthy
// container), nor is a lone vision tower.
func ModelLoadable(e catalog.Entry, ok bool) bool {
	return ok && e.Kind == catalog.KindContainer && e.Probed && !e.Stale && e.ProbeErr == "" &&
		!entryTowerOnly(e)
}

// ModelConvertible reports whether the selected entry is something to convert.
func ModelConvertible(e catalog.Entry, ok bool) bool {
	return ok && (e.Kind == catalog.KindGGUF || e.Kind == catalog.KindSafetensors)
}

// ModelReconvertible reports whether Reconvert applies to the selected entry:
// any container except a lone vision tower, which rebuilt is still a lone
// vision tower.
func ModelReconvertible(e catalog.Entry, ok bool) bool {
	return ok && e.Kind == catalog.KindContainer && !entryTowerOnly(e)
}

// entryTowerOnly reports a container that holds only a vision tower -- an
// mmproj converted by itself.
func entryTowerOnly(e catalog.Entry) bool {
	return e.Kind == catalog.KindContainer && e.HasVision && e.NVocab == 0
}

// SortModels orders the table. An unknown or empty column leaves the scan's
// own order, which is kind then name.
func SortModels(out []catalog.Entry, col string, asc bool) {
	less := func(i, j int) bool { return false }
	switch col {
	case "name":
		less = func(i, j int) bool {
			return strings.ToLower(out[i].Name) < strings.ToLower(out[j].Name)
		}
	case "quant":
		less = func(i, j int) bool { return out[i].Quant < out[j].Quant }
	case "vision":
		less = func(i, j int) bool { return !out[i].HasVision && out[j].HasVision }
	case "fits":
		less = func(i, j int) bool { return out[i].Size < out[j].Size } // fitting is a size question
	case "size":
		less = func(i, j int) bool { return out[i].Size < out[j].Size }
	case "status":
		less = func(i, j int) bool { return entryShortStatus(out[i]) < entryShortStatus(out[j]) }
	default:
		return
	}
	if asc {
		sort.SliceStable(out, less)
		return
	}
	sort.SliceStable(out, func(i, j int) bool { return less(j, i) })
}

// entryShortStatus is the table's verdict, and it names what the row needs.
// It stays short enough for the Status column; the detail pane carries the
// versions.
func entryShortStatus(e catalog.Entry) string {
	switch {
	case e.Kind != catalog.KindContainer:
		return "convert it"
	case !e.Probed:
		return "reading..."
	case e.Stale:
		return "reconvert it"
	case e.ProbeErr != "":
		return "unreadable"
	case entryTowerOnly(e):
		return "vision tower"
	}
	return "ready"
}

// entryHint is the one sentence the detail pane gives a row that cannot be
// loaded, saying why; a loadable row gets none.
func entryHint(e catalog.Entry) string {
	switch {
	case e.Kind != catalog.KindContainer:
		return "The app loads .jlm files, so convert this one first."
	case !e.Probed:
		return ""
	case e.Stale && e.Version > catalog.CurrentVersion:
		return "This file was made by a newer version of the app. Update the app to load it."
	case e.Stale:
		return "This file was made by an older version of the app, so it will not load until it is reconverted."
	case e.ProbeErr != "":
		return "This file cannot be read as a model."
	case entryTowerOnly(e):
		return "This file holds only a vision tower, which cannot be loaded on its own."
	}
	return ""
}

func entryStatus(e catalog.Entry) string {
	switch {
	case e.Kind != catalog.KindContainer:
		return "needs converting"
	case !e.Probed:
		return "reading..."
	case e.Stale:
		return fmt.Sprintf("needs reconverting (file format v%d, this app reads v%d)",
			e.Version, catalog.CurrentVersion)
	case e.ProbeErr != "":
		return "unreadable: " + e.ProbeErr
	case entryTowerOnly(e):
		return "vision tower only"
	}
	return "ready to load"
}

// entryRead reports a container whose probe read a model's config.
func entryRead(e catalog.Entry) bool { return e.Probed && e.NLayer > 0 }

func entryLayers(e catalog.Entry) string {
	if e.NLayer == 0 {
		return "--"
	}
	return fmt.Sprintf("%d", e.NLayer)
}

func entryGeometry(e catalog.Entry) string {
	if e.NLayer == 0 {
		return "--"
	}
	return fmt.Sprintf("%d layers, d %d, ctx %d, vocab %d", e.NLayer, e.NEmbd, e.NCtx, e.NVocab)
}

func entryMixture(e catalog.Entry) string {
	if !e.MoE() {
		if entryRead(e) {
			return "dense"
		}
		return "--"
	}
	return fmt.Sprintf("%d experts, %d routed per token", e.NExpert, e.NExpertUsed)
}

func entryVision(e catalog.Entry) string {
	if !e.HasVision {
		if entryRead(e) {
			return "none"
		}
		return "--"
	}
	return fmt.Sprintf("%d block(s) of %s", e.VisionBlocks, app.Bytes(e.VisPageSize))
}

func entryChat(e catalog.Entry) string {
	if !entryRead(e) {
		return "--"
	}
	if e.HasChat {
		return "yes -- chat mode is available for this model"
	}
	return "no -- completion only"
}

// entryPages is the number that decides whether a model pages at all. A page
// is a whole block, fixed at conversion, and a budget one page short is a
// cliff rather than a gradient.
func entryPages(e catalog.Entry) string {
	if e.NBlocks == 0 {
		return "--"
	}
	return fmt.Sprintf("%d block(s) of %s = %s resident when it all fits",
		e.NBlocks, app.Bytes(e.PageSize), app.Bytes(uint64(e.NBlocks)*e.PageSize))
}

// entryChunk is the pager's read granularity, which is not its page size: a
// page is what is resident, a chunk is what a fault fetches.
func entryChunk(e catalog.Entry) string {
	if e.ChunkBytes == 0 {
		return "--"
	}
	return app.Bytes(e.ChunkBytes) + " per read"
}

func entryStream(e catalog.Entry) string {
	if !entryRead(e) {
		return "--"
	}
	if e.StreamGroups == 0 {
		return "0 -- nothing to stream"
	}
	return fmt.Sprintf("%d", e.StreamGroups)
}

func entryDense(e catalog.Entry) string {
	if e.DenseBytes == 0 {
		return "--"
	}
	return app.Bytes(e.DenseBytes) + " always resident"
}

func modelsNonNeg(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

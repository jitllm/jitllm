package screen

import (
	"path/filepath"
	"strings"

	"github.com/gogpu/ui/core/button"
	"github.com/gogpu/ui/core/progressbar"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/common/catalog"
	"github.com/jitllm/jitllm/common/convertjob"
	"github.com/jitllm/jitllm/common/discover"
	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/widgets"
)

// The Convert screen: the GGUF and safetensors files in the model folders, and
// the form that turns one into a .jlm container. It is its own tab so sources
// and containers are never in one list; the scan is still one walk, owned by
// Models, which publishes the containers there and the sources here.

// convertScreen is this screen's state: the form, its in-flight flag and the
// table's sort. Keyed by shell for the reason modelsScreen is.
type convertScreen struct {
	sh    *app.Shell
	store *app.Store

	src        state.Signal[string]
	mmproj     state.Signal[string]
	dst        state.Signal[string]
	converting state.Signal[bool]
	// note is one sentence under the form about what it holds. Empty takes no
	// space.
	note state.Signal[string]

	convertDisabled state.ReadonlySignal[bool]
}

// convertFor returns this shell's Convert state, creating it once. It starts
// the scan if the Models screen has not, since that one walk fills both lists.
func convertFor(sh *app.Shell) *convertScreen {
	c, created := app.Local(sh, func() *convertScreen {
		c := &convertScreen{
			sh:         sh,
			store:      sh.Store,
			src:        state.NewSignal(""),
			mmproj:     state.NewSignal(""),
			dst:        state.NewSignal(""),
			converting: state.NewSignal(false),
			note:       state.NewSignal(""),
		}
		c.convertDisabled = state.NewComputed(func() bool { return c.converting.Get() }, c.converting.AsReadonly())
		return c
	})
	if created {
		modelsFor(sh)
	}
	return c
}

// Convert builds the Convert screen. It is an [app.ScreenFunc].
func Convert(sh *app.Shell) widget.Widget {
	c := convertFor(sh)
	m := modelsFor(sh)
	p := sh.P
	add := button.New(button.TextOpt("Add folder"), button.SizeOpt(button.Small),
		button.VariantOpt(button.Outlined), button.OnClick(m.addFolder), button.PainterOpt(p.Button))
	rescan := button.New(button.TextOpt("Rescan"), button.SizeOpt(button.Small),
		button.VariantOpt(button.Outlined), button.OnClick(m.rescan), button.PainterOpt(p.Button))
	head := pageHead(sh, "Convert",
		"GGUF and safetensors files become a .jlm the app runs. Each is converted once; the result appears on Models.",
		add, rescan)
	return listPage(sh, head, c.list(), c.formSide(), convertPanelWidth)
}

// convertPanelWidth is the form beside the list: three paths and their Browse
// buttons want more room than Discover's panel.
const convertPanelWidth = 380

// --- the list -----------------------------------------------------------------

// list is one row per source file: what it is, and whether its container
// exists yet. Only a person's click fills the form, so a rescan republishing
// the rows never overwrites an edited one.
func (c *convertScreen) list() widget.Widget {
	st := c.store
	col := widgets.NewColumn(st.SourcesRows.AsReadonly(), 2, func(i int) widget.Widget {
		at := func() (catalog.Entry, bool) {
			if all := st.Sources.Get(); i < len(all) {
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
			}, st.Sources.AsReadonly(), st.Machine.AsReadonly())
		}
		look := state.NewComputed(func() widgets.TileLook {
			e, _ := at()
			return tileLook(c.sh.P, e.Name)
		}, st.Sources.AsReadonly())
		sel := state.NewComputed(func() bool { return st.SourcesSel.Get() == i }, st.SourcesSel.AsReadonly())
		return listRow(c.sh, look,
			text(func(e catalog.Entry) string { return strings.TrimSuffix(e.Name, filepath.Ext(e.Name)) }),
			text(convertjob.SourceFacts),
			text(convertjob.SourceState),
			text(func(e catalog.Entry) string {
				if e.Tower {
					return ""
				}
				return discover.FitWordsOf(discover.EntryFits(e, st.Machine.Get()))
			}),
			sel, func() {
				st.SourcesSel.Set(i)
				c.pick(i)
			})
	})
	none := state.NewComputed(func() bool { return st.SourcesRows.Get() == 0 }, st.SourcesRows.AsReadonly())
	empty := widgets.Hide(none, panel(c.sh,
		c.sh.P.Role(primitives.Text("Nothing to convert").Color(c.sh.P.Text()), c.sh.P.Type.TitleMedium).Bold(),
		wrapped(c.sh, state.NewSignal("No GGUF or safetensors files in your model folders. Browse for one in the form, "+
			"or download a model from Discover, which converts it for you.").AsReadonly()),
	))
	return primitives.VBox(empty, col).CrossAlign(primitives.CrossAxisStretch)
}

// publishSources hands the Convert list a copy sorted by name, keeping the
// selection on the same file, for the reason [modelsScreen.publish] gives.
func publishSources(st *app.Store, found []catalog.Entry) {
	out := make([]catalog.Entry, len(found))
	copy(out, found)
	SortModels(out, "name", true)
	sel := ""
	if e, ok := st.SelectedSource(); ok {
		sel = e.Path
	}
	st.SetSources(out)
	if sel != "" {
		st.SourcesSel.Set(rowOf(out, sel))
	}
}

// pick puts the chosen row in the form: a model as the source, a vision tower
// as the tower. Nothing moves while a conversion runs.
func (c *convertScreen) pick(row int) {
	all := c.store.Sources.Get()
	if row < 0 || row >= len(all) || c.converting.Get() {
		return
	}
	e := all[row]
	if e.Tower {
		c.mmproj.Set(e.Path)
		c.note.Set("")
		return
	}
	c.fillSource(e.Path)
}

// --- the form ---------------------------------------------------------------

func (c *convertScreen) formSide() widget.Widget {
	p := c.sh.P
	bar := primitives.HBox(primitives.Expanded(progressbar.New(
		progressbar.ValueReadonlySignal(c.store.Progress.AsReadonly()),
		progressbar.ColorSchemeOpt(p.ProgressBar),
		progressbar.Height(6),
	)))
	return panel(c.sh,
		p.Role(primitives.Text("Convert a model").Color(p.Text()), p.Type.TitleLarge).Bold(),
		wrapped(c.sh, state.NewSignal("Pick a file in the list or browse for one. For a vision model, add its mmproj file as the vision tower.").AsReadonly()),
		c.pathRow("Source", c.src, placeholderSource, true),
		c.pathRow("Vision tower", c.mmproj, placeholderTower, false),
		c.pathRow("Destination", c.dst, placeholderDest, false),
		hintBox(p, c.note.AsReadonly()),
		primitives.Box().Height(p.Space.XS),
		primitives.HBox(button.New(
			button.TextOpt("Convert"),
			button.SizeOpt(button.Large),
			button.VariantOpt(button.Filled),
			button.OnClick(c.convert),
			button.DisabledReadonlySignal(c.convertDisabled),
			button.PainterOpt(p.Button),
		)),
		widgets.Hide(c.converting.AsReadonly(), primitives.VBox(
			bar,
			widgets.NewParagraph("").ContentSignal(c.store.ProgressLabel).
				FontSize(p.Type.BodySmall.FontSize).Color(p.Text()).Font(p.Prose()).MaxLines(2),
		).Gap(p.Space.XS).CrossAlign(primitives.CrossAxisStretch)),
		wrapped(c.sh, state.NewSignal("The bar is an estimate. A conversion cannot be cancelled once it starts.").AsReadonly()),
	)
}

// The convert form's placeholders. They are short because a textfield paints
// its placeholder at full width (UPSTREAM.md #8);
// TestConvertPlaceholdersFitTheirFields measures them against the field.
const (
	placeholderSource = "GGUF or folder"
	placeholderTower  = "optional mmproj"
	placeholderDest   = "same name, .jlm"
)

// pathRow is one path in the form: the file's name, its folder under it, and
// Browse. It shows the path rather than editing it, because a text field here
// paints a long path straight past the card (UPSTREAM.md #8); the list, Browse
// and a drop onto the window are how a path gets in. An optional path gets a
// way back out.
func (c *convertScreen) pathRow(label string, sig state.Signal[string], hint string, alsoDest bool) widget.Widget {
	p := c.sh.P
	set := state.NewComputed(func() bool { return sig.Get() != "" }, sig.AsReadonly())
	unset := state.NewComputed(func() bool { return sig.Get() == "" }, sig.AsReadonly())
	name := state.NewComputed(func() string { return filepath.Base(sig.Get()) }, sig.AsReadonly())
	dir := state.NewComputed(func() string { return filepath.Dir(sig.Get()) }, sig.AsReadonly())
	shown := primitives.VBox(
		widgets.Hide(set, widgets.NewParagraph("").ContentSignal(name).FontSize(p.Type.BodyMedium.FontSize).
			Color(p.Text()).MaxLines(1).Font(p.Prose())),
		widgets.Hide(set, widgets.NewParagraph("").ContentSignal(dir).FontSize(p.Type.LabelSmall.FontSize).
			Color(p.Muted()).MaxLines(1).Font(p.Mono())),
		widgets.Hide(unset, widgets.NewParagraph(hint).FontSize(p.Type.BodyMedium.FontSize).
			Color(p.Muted()).MaxLines(1).Font(p.Prose())),
	).Gap(2).CrossAlign(primitives.CrossAxisStretch)
	row := []widget.Widget{primitives.Expanded(shown)}
	if label == "Vision tower" {
		row = append(row, widgets.Hide(set, widgets.NewIconButton(widgets.IconTrash(), "No vision tower",
			func() { sig.Set("") }, iconColors(c.sh))))
	}
	row = append(row, button.New(
		button.TextOpt("Browse"),
		button.VariantOpt(button.Outlined),
		button.OnClick(func() { c.browse(label, sig, alsoDest) }),
		button.PainterOpt(p.Button),
	))
	return primitives.VBox(
		p.Role(primitives.Text(label).Color(p.Muted()), p.Type.LabelMedium),
		primitives.Box(primitives.HBox(row...).Gap(p.Space.S)).
			PaddingXY(p.Space.M, p.Space.S).
			Rounded(p.Shape.Medium).
			BorderStyle(1, p.Colors.OutlineVariant).
			Background(p.Colors.SurfaceContainerLow),
	).Gap(p.Space.XS).CrossAlign(primitives.CrossAxisStretch)
}

// fillSource puts path in the form with its destination derived, and says
// what converting it would replace.
func (c *convertScreen) fillSource(path string) {
	c.src.Set(path)
	c.mmproj.Set("")
	dst := DestFor(path, primaryDir(c.sh.Cfg.ModelDirs))
	c.dst.Set(dst)
	c.note.Set(convertjob.ReplaceNote(dst))
	c.sh.SetStatus("convert source: " + filepath.Base(path) + " -- press Convert")
}

// selectSource highlights path's row in the Convert table, if it lists one.
func (c *convertScreen) selectSource(path string) bool {
	i := rowOf(c.store.Sources.Get(), path)
	if i < 0 {
		return false
	}
	c.store.SourcesSel.Set(i)
	return true
}

func (c *convertScreen) browse(title string, into state.Signal[string], alsoDest bool) {
	start := ""
	if len(c.sh.Cfg.ModelDirs) > 0 {
		start = c.sh.Cfg.ModelDirs[0]
	}
	paths := c.sh.PickFile(title, modelFilters(), start)
	if len(paths) == 0 {
		return
	}
	into.Set(paths[0])
	if alsoDest {
		c.dst.Set(DestFor(paths[0], primaryDir(c.sh.Cfg.ModelDirs)))
	}
	c.note.Set("")
}

func (c *convertScreen) convert() {
	req := ConvertRequest{Src: c.src.Get(), MMProj: c.mmproj.Get(), Dst: c.dst.Get(),
		Dir: primaryDir(c.sh.Cfg.ModelDirs)}
	plan, err := PlanConvert(req)
	if err != nil {
		c.sh.Alert("Cannot convert", err.Error())
		return
	}
	c.note.Set("")
	c.dst.Set(plan.Dst)
	c.converting.Set(true)
	c.sh.SetStatus("converting " + filepath.Base(plan.Src))

	m := modelsFor(c.sh)
	took := m.runner()(func() {
		defer c.converting.Set(false)
		if err := RunConvert(c.sh, plan); err != nil {
			c.sh.Alert("Convert failed", err.Error())
			c.sh.SetStatus("convert failed: " + err.Error())
			return
		}
		// Back to the UI goroutine: rescan snapshots the configured
		// directories, and the config belongs to that goroutine.
		c.sh.Post(m.rescan)
	})
	// A dropped job never runs its defer, so clear the flag here.
	if !took {
		c.converting.Set(false)
		c.sh.SetStatus("the engine is busy: press Convert again in a moment")
	}
}

// reconvertPath rebuilds the container at path from the GGUF beside it, in one
// click when that is safe, and otherwise fills the form and says what is
// missing; [convertjob.PlanReconvert] decides which. OfferReconvert is not a
// button, so the running flag is checked there too: two conversions at once
// would fight over one progress bar.
func (c *convertScreen) reconvertPath(path string) {
	r := convertjob.PlanReconvert(path, c.converting.Get())
	c.src.Set(r.Req.Src)
	c.mmproj.Set("")
	c.dst.Set(r.Req.Dst)
	c.note.Set(r.Note)
	c.selectSource(r.Req.Src)
	if r.Status != "" {
		c.sh.SetStatus(r.Status)
	}
	if r.Start {
		c.convert()
	}
}

// OfferConvert switches to the Convert tab with the form filled from path, and
// selects path's row if the table lists it. Call it on the UI goroutine.
func OfferConvert(sh *app.Shell, path string) {
	c := convertFor(sh)
	sh.GoConvert()
	c.selectSource(path)
	c.fillSource(path)
}

// OfferReconvert switches to the Convert tab and runs the reconvert flow on the
// container at path: the conversion starts at once when its GGUF is beside it
// and the file cannot load, and otherwise the form is filled and says what to
// pick. Call it on the UI goroutine.
func OfferReconvert(sh *app.Shell, path string) {
	c := convertFor(sh)
	sh.GoConvert()
	c.reconvertPath(path)
}

// hintBox is a one-sentence callout bound to sig. It takes no space while sig
// is empty. It is a filled box rather than a left rule, because an HBox gives
// its children no minimum height and a rule would measure zero tall.
func hintBox(p app.Painters, sig state.ReadonlySignal[string]) widget.Widget {
	body := widgets.NewParagraph("").
		ContentSignal(sig).
		FontSize(p.Type.BodySmall.FontSize).
		LineHeight(p.Type.BodySmall.LineHeight / p.Type.BodySmall.FontSize).
		Color(p.Text()).
		Font(p.Prose())
	shown := state.NewComputed(func() bool { return sig.Get() != "" }, sig)
	return widgets.Hide(shown, primitives.Box(
		primitives.Box(body).
			PaddingXY(p.Space.S, p.Space.XS).
			Background(p.Colors.SurfaceContainerHigh).
			Rounded(p.Shape.ExtraSmall),
	).PaddingXY(0, p.Space.XS))
}

package server

import (
	"errors"
	"fmt"
	"io/fs"
	"path/filepath"
	"slices"
	"sort"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/internal/localserver"
)

// The models folder is the server's library: every container in it can be
// named by a request, by its file name (with or without .jlm), by an alias,
// or once loaded by its id, and a request naming one that is not loaded loads
// it (Config.OnDemand). The folder is read again at every lookup, so a model
// converted into it is there for the next request without a restart.
//
// What a person sets about a model -- its aliases, whether it is enabled --
// lives beside the models in StateFile, keyed by file name, so it moves with
// the folder and survives a restart.

// The file's format is localserver's (localserver.StateFile), which `jitllm
// run` and `jitllm convert` read and write too.
type (
	modelState   = localserver.ModelState
	libraryState = localserver.State
)

// library is the folder's containers with their settings and load state.
func (e *Engine) library() ([]ModelFileInfo, libraryState, error) {
	files, _, err := e.ScanModels("")
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return nil, libraryState{}, err
	}
	st, err := localserver.ReadState(e.ModelDir())
	if err != nil {
		return nil, st, err
	}
	out := files[:0]
	for _, f := range files {
		if f.IsContainer {
			out = append(out, f)
		}
	}
	return out, st, nil
}

// matches reports whether name names f: its file name, its stem or an alias.
func matches(f ModelFileInfo, name string) bool {
	return localserver.Names(filepath.Base(f.Path), f.Aliases, name)
}

// findFile is the container name names in the models folder.
func (e *Engine) findFile(name string) (ModelFileInfo, bool, error) {
	files, _, err := e.library()
	if err != nil {
		return ModelFileInfo{}, false, err
	}
	for _, f := range files {
		if matches(f, name) {
			return f, true, nil
		}
	}
	return ModelFileInfo{}, false, nil
}

// aliasesOf is the aliases of the container at path, for a loaded model.
func (e *Engine) aliasesOf(path string) []string {
	if filepath.Dir(path) != filepath.Clean(e.ModelDir()) {
		return nil
	}
	st, err := localserver.ReadState(e.ModelDir())
	if err != nil {
		return nil
	}
	if m := st.Models[filepath.Base(path)]; m != nil {
		return m.Aliases
	}
	return nil
}

// loadedNamed is the loaded model name names: its id, its file name, or an
// alias of its file.
func (e *Engine) loadedNamed(name string) *LoadedModel {
	if lm, err := e.Model(name); err == nil {
		return lm
	}
	for _, lm := range e.Models() {
		if lm.name == name || filepath.Base(lm.path) == name || slices.Contains(e.aliasesOf(lm.path), name) {
			return lm
		}
	}
	return nil
}

// resolveModel takes what a request named -- a loaded model's id, a file
// name in the models folder (with or without .jlm), or an alias -- and
// returns the loaded model, loading an enabled one on demand.
func (e *Engine) resolveModel(name string) (*LoadedModel, error) {
	if name == "" {
		return nil, fmt.Errorf("%w: model is required", ErrInvalid)
	}
	if lm := e.loadedNamed(name); lm != nil {
		return lm, nil
	}
	if e.cfg.NoOnDemand {
		return nil, fmt.Errorf("%w: no loaded model %q: load it first, or call ListModels", ErrNotFound, name)
	}
	f, ok, err := e.findFile(name)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("%w: no model %q is loaded or in the models folder %s "+
			"(by id, file name or alias); `jitllmd models` lists them", ErrNotFound, name, e.ModelDir())
	}
	if f.Disabled {
		return nil, fmt.Errorf("%w: model %q (%s) is disabled, so it is not loaded on demand; "+
			"enable it with `jitllmd models -enable %s`, or load it explicitly", ErrInvalid, name, filepath.Base(f.Path), localserver.Stem(filepath.Base(f.Path)))
	}
	e.demandMu.Lock()
	defer e.demandMu.Unlock()
	// Another request may have loaded it while this one waited.
	if lm := e.loadedNamed(name); lm != nil {
		return lm, nil
	}
	for _, lm := range e.Models() {
		if lm.path == f.Path {
			return lm, nil
		}
	}
	o := e.cfg.OnDemand
	o.Path = f.Path
	o.ModelID = localserver.Stem(filepath.Base(f.Path))
	if _, err := e.Model(o.ModelID); err == nil {
		o.ModelID = "" // a model loaded from elsewhere holds the id: take a fresh one
	}
	lm, err := e.LoadModel(o)
	if err != nil {
		budget := o.PageBudgetBytes
		if budget == 0 {
			budget = sched.MemBudget()
		}
		return nil, &LoadError{Model: name, Path: f.Path, Size: f.Size, Budget: budget, Err: err}
	}
	return lm, nil
}

// LoadError is a load on demand that failed: the request named a model that
// is in the folder and enabled, and opening it did not succeed. It names the
// model, its size and the host budget it was opened against.
type LoadError struct {
	Model, Path  string
	Size, Budget uint64
	Err          error
}

func (e *LoadError) Error() string {
	return fmt.Sprintf("loading model %q (%s, %s) on demand within a host budget of %s: %v",
		e.Model, e.Path, humanBytes(e.Size), humanBytes(e.Budget), e.Err)
}

func (e *LoadError) Unwrap() error { return e.Err }

// SetModelOptions adds and removes aliases of a container in the models
// folder and enables or disables it, persisting the change. An alias is
// unique: it may not be another model's alias, file name or stem, nor a
// loaded model's id.
func (e *Engine) SetModelOptions(name string, add, remove []string, enabled *bool) (ModelFileInfo, error) {
	e.libMu.Lock()
	defer e.libMu.Unlock()
	if lm := e.loadedNamed(name); lm != nil && filepath.Dir(lm.path) == filepath.Clean(e.ModelDir()) {
		name = filepath.Base(lm.path)
	}
	files, st, err := e.library()
	if err != nil {
		return ModelFileInfo{}, err
	}
	var f *ModelFileInfo
	for i := range files {
		if matches(files[i], name) {
			f = &files[i]
			break
		}
	}
	if f == nil {
		return ModelFileInfo{}, fmt.Errorf("%w: no model %q in the models folder %s", ErrNotFound, name, e.ModelDir())
	}
	key := filepath.Base(f.Path)
	for _, a := range add {
		if err := localserver.ValidAlias(a); err != nil {
			return ModelFileInfo{}, fmt.Errorf("%w: %v", ErrInvalid, err)
		}
		for _, g := range files {
			if g.Path != f.Path && matches(g, a) {
				return ModelFileInfo{}, fmt.Errorf("%w: alias %q already names %s", ErrExists, a, filepath.Base(g.Path))
			}
		}
		for _, lm := range e.Models() {
			if lm.id == a && lm.path != f.Path {
				return ModelFileInfo{}, fmt.Errorf("%w: alias %q is the id of loaded model %s", ErrExists, a, lm.path)
			}
		}
	}
	m := st.Models[key]
	if m == nil {
		m = &modelState{}
		st.Models[key] = m
	}
	for _, a := range add {
		if !slices.Contains(m.Aliases, a) && a != key && a != localserver.Stem(key) {
			m.Aliases = append(m.Aliases, a)
		}
	}
	m.Aliases = slices.DeleteFunc(m.Aliases, func(a string) bool { return slices.Contains(remove, a) })
	sort.Strings(m.Aliases)
	if enabled != nil {
		m.Disabled = !*enabled
	}
	out := *f
	out.Aliases, out.Disabled = slices.Clone(m.Aliases), m.Disabled
	if err := localserver.WriteState(e.ModelDir(), st); err != nil {
		return ModelFileInfo{}, err
	}
	return out, nil
}

// ListAvailable is every container in the models folder that is not loaded
// and not disabled, for /v1/models: a request may name any of them.
func (e *Engine) ListAvailable() []ModelSummary {
	if e.cfg.NoOnDemand {
		return nil
	}
	files, _, err := e.library()
	if err != nil {
		return nil
	}
	out := []ModelSummary{}
	for _, f := range files {
		if f.Disabled || f.ModelID != "" {
			continue
		}
		out = append(out, ModelSummary{ID: localserver.Stem(filepath.Base(f.Path)), Name: f.Name, Aliases: f.Aliases})
	}
	return out
}

// applyState fills a scan's settings and load state from dir's state file.
func (e *Engine) applyState(dir string, files []ModelFileInfo) {
	st, _ := localserver.ReadState(dir)
	loaded := map[string]string{}
	for _, lm := range e.Models() {
		loaded[filepath.Clean(lm.path)] = lm.id
	}
	for i := range files {
		if m := st.Models[filepath.Base(files[i].Path)]; m != nil {
			files[i].Aliases, files[i].Disabled = slices.Clone(m.Aliases), m.Disabled
		}
		files[i].ModelID = loaded[filepath.Clean(files[i].Path)]
	}
}

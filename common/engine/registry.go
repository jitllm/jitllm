package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/server"
)

// entry is one model the app loaded into the engine. Its tier, opened with
// the device choice at load (spec), stays with it while another is in use;
// the chat's session is the active entry's alone.
type entry struct {
	m    *model.Model
	lm   *server.LoadedModel
	path string
	spec string
	// placement is the -placement map the model was opened under; a load under
	// another re-opens it, as another spec does.
	placement string
}

// find returns the open entry for path, or nil.
func (e *Engine) find(path string) *entry {
	for _, en := range e.models {
		if en.path == path {
			return en
		}
	}
	return nil
}

// publishModels mirrors the registry into the store.
func (e *Engine) publishModels() {
	out := make([]session.LoadedModel, 0, len(e.models))
	for i, en := range e.models {
		lm := session.LoadedModel{
			Path:   en.path,
			Name:   short(en.path),
			Colour: i,
			Grant:  en.lm.Budget(),
			Pinned: en.lm.Pinned(),
			Active: e.active == en,
		}
		if en.m != nil {
			lm.Arch = en.m.Cfg.Arch
			lm.Blocks = en.m.Cfg.NLayer
		}
		out = append(out, lm)
	}
	active := ""
	if e.active != nil {
		active = e.active.path
	}
	e.sh.Post(func() {
		e.st.Models.Set(out)
		e.st.Active.Set(active)
	})
}

// Use makes an already-open model the active one. Loading it is Load's job.
func (e *Engine) Use(path string) {
	if _, ok := session.LoadingOf(e.st.Loads, path); ok {
		return
	}
	e.busyBegin()
	session.QueueLoad(e.st.Loads, session.Loading{Path: path, Stage: "Switching models"})
	ok := e.post("use", func() {
		defer e.busyEnd()
		defer e.loading(path)()
		en := e.find(path)
		if en == nil {
			e.status("%s is not open", short(path))
			return
		}
		if e.active == en {
			return
		}
		e.status("switching to %s", short(path))
		if err := e.activate(en); err != nil {
			e.report(stageDevice, path, err)
		}
	})
	if !ok {
		e.busyEnd()
		session.EndLoad(e.st.Loads, path)
	}
}

// Unload closes one model. Closing the active one leaves nothing running.
func (e *Engine) Unload(path string) {
	e.post("unload", func() {
		en := e.find(path)
		if en == nil {
			return
		}
		e.unload(en)
		e.publishModels()
		if e.active == nil {
			e.sh.Post(func() { e.st.Loaded.Set(false); e.st.Vision.Set(false) })
		}
		e.status("closed %s", short(path))
	})
}

// unload closes en's model, and the chat's session with it when it is the
// active one. Forced: an API session on the model closes with it. The
// remaining models grow into the bytes it held (server.Engine.UnloadModel).
func (e *Engine) unload(en *entry) {
	if e.active == en {
		e.releaseSession()
		e.active = nil
	}
	if _, err := e.srv.UnloadModel(en.lm.ID(), true); err != nil {
		e.status("closing %s: %v", short(en.path), err)
	}
	for i, x := range e.models {
		if x == en {
			e.models = append(e.models[:i], e.models[i+1:]...)
			break
		}
	}
}

// Pin caps one model's host budget, or releases the cap at 0.
func (e *Engine) Pin(path string, bytes uint64) {
	e.post("pin", func() {
		en := e.find(path)
		if en == nil {
			return
		}
		if err := e.srv.Pin(en.lm.ID(), bytes); err != nil {
			e.status("%s: %v", short(path), err)
			return
		}
		e.publishModels()
		if bytes == 0 {
			e.status("%s: budget divided automatically again", short(path))
			return
		}
		e.status("%s: capped at %s", short(path), session.Bytes(bytes))
	})
}

// SetPriority toggles whether the active model gets the lion's share.
func (e *Engine) SetPriority(on bool) {
	e.post("priority", func() {
		e.srv.SetPriority(on)
		e.publishModels()
		if on {
			e.status("the model in chat gets the memory; chatting with another moves it")
			return
		}
		e.status("memory divided in load order")
	})
}

// errNoModel is returned when a switch is asked for with nothing to switch to.
var errNoModel = fmt.Errorf("no model is open")

// activeName is the running model's display name and palette index, for the
// turn it is about to produce.
func (e *Engine) activeName() (string, int) {
	if e.active == nil {
		return "", 0
	}
	for i, en := range e.models {
		if en == e.active {
			return short(en.path), i
		}
	}
	return short(e.active.path), 0
}

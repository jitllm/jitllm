package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/jit_old/gpu/tier"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// cmdModels is ModelService: what is on disk, what is loaded, and the two
// mutations that move a model between those states (`models -load` rather
// than a separate verb, since it is the same service).
func cmdModels(ctx context.Context, c cli, args []string) error {
	fs := flag.NewFlagSet("jitllmd models", flag.ContinueOnError)
	addr := addrFlag(fs)
	dir := fs.String("dir", "", "directory to scan (default: the server's own model directory)")
	loadedOnly := fs.Bool("loaded", false, "list only models that are currently loaded")

	load := fs.String("load", "", "a .jlm container to load, absolute or relative to the model directory")
	id := fs.String("id", "", "model id for -load (default: derived)")
	devices := fs.String("devices", "",
		"devices for -load: auto|all|cpu|gpu[:N]|(cuda|vulkan|metal)[:DEVICE][=BYTES], comma-separated; "+
			"a bare gpu, cuda or vulkan is every such device, a :DEVICE selector pins one")
	maxmem := fs.String("maxmem", "", "page budget for -load, e.g. 8G")
	gpuLayers := fs.Int("gpu-layers", -1, "blocks to place on a device for -load; -1 is as many as fit")
	kvF16 := fs.Bool("kv-f16", false, "force an f16 KV cache for -load (default: the engine's per-tier choice)")

	unload := fs.String("unload", "", "a loaded model id to unload")
	force := fs.Bool("force", false, "with -unload, close sessions holding the model instead of refusing")

	alias := fs.String("alias", "", "MODEL=ALIAS[,ALIAS...]: other names a request may use for a model in the models folder")
	unalias := fs.String("unalias", "", "MODEL=ALIAS[,ALIAS...]: remove aliases")
	enable := fs.String("enable", "", "a model in the models folder a request may load on demand (the default)")
	disable := fs.String("disable", "", "a model in the models folder that is listed and never loaded on demand")

	if err := parse(c, fs, args); err != nil {
		return err
	}
	cn, err := target(ctx, c, fs, *addr)
	if err != nil {
		return err
	}

	if *alias != "" || *unalias != "" || *enable != "" || *disable != "" {
		return setModelOptions(ctx, c, cn, *alias, *unalias, *enable, *disable)
	}

	switch {
	case *load != "":
		return loadModel(ctx, c, cn, loadOpts{
			path: *load, id: *id, devices: *devices, maxmem: *maxmem,
			gpuLayers: *gpuLayers, gpuLayersSet: wasSet(fs, "gpu-layers"),
			kvF16: *kvF16, kvF16Set: wasSet(fs, "kv-f16"),
		})
	case *unload != "":
		resp, err := cn.Model.UnloadModel(ctx, connect.NewRequest(&v1.UnloadModelRequest{
			ModelId: *unload, Force: *force,
		}))
		if err != nil {
			return clientError("unloading "+*unload, cn.base, err)
		}
		fmt.Fprintf(c.out, "unloaded %s, %d session(s) closed\n", *unload, resp.Msg.GetSessionsClosed())
		return nil
	}

	if name := strings.Join(fs.Args(), " "); name != "" {
		resp, err := cn.Model.GetModel(ctx, connect.NewRequest(&v1.GetModelRequest{ModelId: name}))
		if err != nil {
			return clientError("describing "+name, cn.base, err)
		}
		describeModel(c.out, resp.Msg.GetModel())
		return nil
	}

	resp, err := cn.Model.ListModels(ctx, connect.NewRequest(&v1.ListModelsRequest{
		Directory: *dir, LoadedOnly: *loadedOnly,
	}))
	if err != nil {
		return clientError("listing models", cn.base, err)
	}
	listModels(c.out, resp.Msg)
	return nil
}

// setModelOptions applies -alias, -unalias, -enable and -disable, one
// SetModelOptions call per model named.
func setModelOptions(ctx context.Context, c cli, cn *conn, alias, unalias, enable, disable string) error {
	type change struct {
		add, remove []string
		enabled     *bool
	}
	changes := map[string]*change{}
	var order []string
	get := func(name string) *change {
		if changes[name] == nil {
			changes[name] = &change{}
			order = append(order, name)
		}
		return changes[name]
	}
	split := func(flag, v string) (string, []string, error) {
		name, list, ok := strings.Cut(v, "=")
		if !ok || name == "" || list == "" {
			return "", nil, usagef("-%s %q: want MODEL=ALIAS[,ALIAS...]", flag, v)
		}
		return name, csv(list), nil
	}
	if alias != "" {
		n, a, err := split("alias", alias)
		if err != nil {
			return err
		}
		get(n).add = a
	}
	if unalias != "" {
		n, a, err := split("unalias", unalias)
		if err != nil {
			return err
		}
		get(n).remove = a
	}
	on, off := true, false
	if enable != "" {
		get(enable).enabled = &on
	}
	if disable != "" {
		if enable == disable {
			return usagef("-enable and -disable name the same model")
		}
		get(disable).enabled = &off
	}
	for _, name := range order {
		ch := changes[name]
		resp, err := cn.Model.SetModelOptions(ctx, connect.NewRequest(&v1.SetModelOptionsRequest{
			Name: name, AddAliases: ch.add, RemoveAliases: ch.remove, Enabled: ch.enabled,
		}))
		if err != nil {
			return clientError("setting "+name, cn.base, err)
		}
		f := resp.Msg.GetFile()
		state := "enabled"
		if f.GetDisabled() {
			state = "disabled"
		}
		fmt.Fprintf(c.out, "%s  %s  aliases: %s\n", f.GetName(), state, aliasList(f.GetAliases()))
	}
	return nil
}

func aliasList(a []string) string {
	if len(a) == 0 {
		return "none"
	}
	return strings.Join(a, ", ")
}

type loadOpts struct {
	path, id, devices, maxmem string
	gpuLayers                 int
	gpuLayersSet              bool
	kvF16, kvF16Set           bool
}

func loadModel(ctx context.Context, c cli, cn *conn, o loadOpts) error {
	req := &v1.LoadModelRequest{Path: o.path, ModelId: o.id, DeviceIds: csv(o.devices)}
	if o.maxmem != "" {
		// tier.ParseBytes, so -maxmem means what it means to `jitllm run`.
		b, err := tier.ParseBytes(o.maxmem)
		if err != nil {
			return usagef("-maxmem: %v", err)
		}
		req.PageBudgetBytes = b
	}
	if o.gpuLayersSet {
		v := int32(o.gpuLayers)
		req.MaxDeviceBlocks = &v
	}
	if o.kvF16Set {
		v := o.kvF16
		req.KvF16 = &v
	}
	resp, err := cn.Model.LoadModel(ctx, connect.NewRequest(req))
	if err != nil {
		// A .gguf comes back as FailedPrecondition carrying the convert
		// command; clientError prints it verbatim.
		return clientError("loading "+o.path, cn.base, err)
	}
	m := resp.Msg.GetModel()
	fmt.Fprintf(c.out, "loaded %s in %s\n", m.GetModelId(),
		(time.Duration(resp.Msg.GetLoadMillis()) * time.Millisecond).Round(time.Millisecond))
	describeModel(c.out, m)
	return nil
}

func listModels(w io.Writer, m *v1.ListModelsResponse) {
	if d := m.GetDirectory(); d != "" {
		fmt.Fprintf(w, "directory %s\n", d)
	}
	loaded := m.GetLoaded()
	fmt.Fprintf(w, "\nloaded (%d)\n", len(loaded))
	for _, lm := range loaded {
		where := "host"
		if ids := lm.GetDefaultDeviceIds(); len(ids) > 0 {
			where = strings.Join(ids, ",")
		}
		fmt.Fprintf(w, "  %-24s %-22s %-10s %3d blocks  %6d ctx  page %-11s  %d session(s)  [%s]\n",
			lm.GetModelId(), lm.GetName(), lm.GetArchitecture(), lm.GetBlockCount(),
			lm.GetContextLength(), bytesOf(lm.GetPageSize()), lm.GetSessionCount(), where)
	}
	files := m.GetFiles()
	if len(files) == 0 {
		return
	}
	fmt.Fprintf(w, "\non disk (%d)\n", len(files))
	for _, f := range files {
		// A .gguf is listed with the command that makes it loadable.
		if f.GetIsContainer() {
			// Version 0 means the header did not parse, not an old version;
			// the container needs re-converting.
			ver := fmt.Sprintf("v%d", f.GetContainerVersion())
			if f.GetContainerVersion() == 0 {
				ver = "header did not parse"
			}
			var notes []string
			if id := f.GetModelId(); id != "" {
				notes = append(notes, "loaded as "+id)
			}
			if f.GetDisabled() {
				notes = append(notes, "disabled")
			}
			if a := f.GetAliases(); len(a) > 0 {
				notes = append(notes, "aliases "+strings.Join(a, ", "))
			}
			note := ""
			if len(notes) > 0 {
				note = "  (" + strings.Join(notes, "; ") + ")"
			}
			fmt.Fprintf(w, "  %-44s %12s  %s%s\n", f.GetName(), bytesOf(f.GetSize()), ver, note)
			continue
		}
		fmt.Fprintf(w, "  %-44s %12s  needs conversion:\n      %s\n",
			f.GetName(), bytesOf(f.GetSize()), f.GetConvertCommand())
	}
}

func describeModel(w io.Writer, m *v1.ModelInfo) {
	if m == nil {
		return
	}
	fmt.Fprintf(w, "\n%s\n", m.GetModelId())
	fmt.Fprintf(w, "  path         %s\n", m.GetPath())
	fmt.Fprintf(w, "  architecture %s\n", m.GetArchitecture())
	fmt.Fprintf(w, "  geometry     %d block(s), %d embd, %d ctx, %d vocab\n",
		m.GetBlockCount(), m.GetEmbeddingDim(), m.GetContextLength(), m.GetVocabSize())
	if m.GetExpertCount() > 0 {
		fmt.Fprintf(w, "  mixture      %d expert(s), %d routed per token\n",
			m.GetExpertCount(), m.GetExpertsUsed())
	}
	// A hybrid's split is printed because relocating the two kinds differs:
	// an attention block carries a KV cache, a linear one a recurrent summary
	// that must be migrated, not dropped.
	if m.GetLinearBlocks() > 0 {
		fmt.Fprintf(w, "  hybrid       %d attention + %d linear (recurrent)\n",
			m.GetAttentionBlocks(), m.GetLinearBlocks())
	}
	if m.GetHasVisionTower() {
		fmt.Fprintf(w, "  vision       %d block(s), page %s\n",
			m.GetVisionBlockCount(), bytesOf(m.GetVisionPageSize()))
	}
	fmt.Fprintf(w, "  pager        page %s, weights %s\n",
		bytesOf(m.GetPageSize()), bytesOf(m.GetWeightBytes()))
	if m.GetChatCapable() {
		fmt.Fprintf(w, "  chat         yes%s\n", templateNames(m.GetChatTemplateNames()))
	} else {
		fmt.Fprintf(w, "  chat         NO -- a -chat request against this model is refused rather "+
			"than run as a raw completion\n")
	}
	where := "host"
	if ids := m.GetDefaultDeviceIds(); len(ids) > 0 {
		where = strings.Join(ids, ",")
	}
	fmt.Fprintf(w, "  sessions     %d, default placement %s\n", m.GetSessionCount(), where)
}

func templateNames(n []string) string {
	if len(n) == 0 {
		return ""
	}
	return " (" + strings.Join(n, ", ") + ")"
}

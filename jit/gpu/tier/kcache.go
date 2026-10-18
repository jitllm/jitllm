package tier

import (
	"crypto/sha256"
	"encoding/gob"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// closeKern closes a scratch's kernel unless it is shared: a kernel
// compileShared handed out stays open when a scratch frees it, since the next
// scratch of the same shape takes the same compiled kernel, and is closed with
// its device (closeShared). Callers hold g.mu, which guards g.kshared.
func (g *devShared) closeKern(k backend.Kernel) {
	if k == nil {
		return
	}
	if _, ok := g.kshared[k]; ok {
		return
	}
	k.Close()
}

// irKey is a kernel's identity by content: the IR, not the name, since a
// scratch kernel's name does not carry every constant baked into it.
func irKey(k *ir.Kernel) (string, error) {
	h := sha256.New()
	if err := gob.NewEncoder(h).Encode(k); err != nil {
		return "", err
	}
	return string(h.Sum(nil)), nil
}

// compileShared is Compile through the device's content-keyed cache, for the
// kernels a batched scratch builds for itself (norms, rotary, attention): a
// scratch of one width dropped and rebuilt -- every prompt width is built once
// at the end of a placement and dropped (prewarmPromptWidths) -- takes its
// kernels back from the cache instead of compiling inside a prompt. Callers
// hold g.mu.
func (g *devTier) compileShared(k *ir.Kernel) (backend.Kernel, error) {
	key, err := irKey(k)
	if err != nil {
		return g.dev.Compile(k)
	}
	if c, ok := g.kshareKey[key]; ok {
		return c, nil
	}
	c, err := g.dev.Compile(k)
	if err != nil {
		return nil, err
	}
	if g.kshareKey == nil {
		g.kshareKey, g.kshared = map[string]backend.Kernel{}, map[backend.Kernel]struct{}{}
	}
	g.kshareKey[key] = c
	g.kshared[c] = struct{}{}
	return c, nil
}

// closeShared closes the device's cached scratch kernels. Callers hold g.mu.
func (g *devTier) closeShared() {
	for key, c := range g.kshareKey {
		delete(g.kshared, c)
		c.Close()
		delete(g.kshareKey, key)
	}
}

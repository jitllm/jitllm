package model

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// Kimi-K3's text model (jlm.ArchKimiK3), transcribed from Moonshot's
// modeling_kimi_linear.py, which llama.cpp's kimi-k3.cpp matches: Kimi-Linear's
// hybrid (KDA in delta.go, MLA with no rotary in mla.go) plus residual
// attention over a bank of block inputs, a latent mixture (k3MoE), situ
// (graph.ActSitu), an MLA output gate (mlaOutProject) and a full-rank KDA gate
// with an optional decay lower bound (nn.DeltaDecayBound32JIT).
//
// The bank crosses a device seam inside a token, so it is carried as residual
// STREAMS (Config.ResidW), stream-major: stream 0 is the running residual and
// stream 1+j checkpoint j. Every n_embd kernel that reads the residual reads
// stream 0 untouched; a device is handed every stream and hands every stream
// back, and the head's mix runs on the host (streamHead).
//
// Each piece's arithmetic: docs/engineering-history/model-correctness.md,
// "model/k3.go".

// k3Fault breaks one piece of Kimi-K3's graph, for a gate's violation only.
type K3Fault int

// halfFault drops one half of a block that attends beside its mixer: the
// half computed and its output weighed by zero, so every tensor it reads
// stays bound as the container stores it.
type HalfFault int

const (
	halfFaultNone HalfFault = iota
	// halfFaultNoAttn leaves the block its mixer alone.
	HalfFaultNoAttn
	// halfFaultNoMixer leaves the block its attention alone.
	HalfFaultNoMixer
)

const (
	k3FaultNone K3Fault = iota
	// k3FaultNoBank mixes nothing: every sublayer reads the running residual,
	// as Kimi-Linear does, and the bank is never read.
	K3FaultNoBank
	// k3FaultNoRestart adds the attention output on a checkpoint block
	// instead of restarting the residual from it.
	K3FaultNoRestart
	// k3FaultRawScores scores the bank on the raw values instead of the
	// RMSNormed ones.
	K3FaultRawScores
	// k3FaultNoHeadMix reads the head from the running residual alone.
	K3FaultNoHeadMix
	// k3FaultNoLatentNorm skips the routed sum's RMSNorm in the latent.
	K3FaultNoLatentNorm
	// k3FaultNoMLAGate leaves the MLA output ungated.
	K3FaultNoMLAGate
	// k3FaultNoKDAGate leaves the linear blocks' output ungated.
	K3FaultNoKDAGate
)

// ResAttn reports Kimi-K3's residual attention: the bank of checkpoints and
// the mixes over it.
func (c *Config) ResAttn() bool { return c.AttnResBlock != 0 }

// resStreams is how many streams the residual attention carries: the
// running residual and one per checkpoint block, or zero without it.
func (c *Config) resStreams() int {
	if !c.ResAttn() {
		return 0
	}
	return 1 + c.resBankFinal()
}

// resBankFinal is how many checkpoints the bank holds after the last block:
// one per block whose index is a multiple of AttnResBlock.
func (c *Config) resBankFinal() int { return (c.NLayer + c.AttnResBlock - 1) / c.AttnResBlock }

// resBank is how many checkpoints block li's attention (ffn false) or FFN
// (ffn true) mixes: those banked before it, and for the FFN its own.
func (c *Config) ResBank(li int, ffn bool) int {
	if ffn {
		return li/c.AttnResBlock + 1
	}
	return (li + c.AttnResBlock - 1) / c.AttnResBlock
}

// resCheckpoint reports whether block li banks its input, and the stream it
// banks into.
func (c *Config) ResCheckpoint(li int) (stream int, ok bool) {
	if !c.ResAttn() || li%c.AttnResBlock != 0 {
		return 0, false
	}
	return 1 + li/c.AttnResBlock, true
}

// ExpWidth is the routed experts' input and output width: ExpertLatent on a
// latent mixture, n_embd everywhere else.
func (c *Config) ExpWidth() int {
	if c.ExpertLatent != 0 {
		return c.ExpertLatent
	}
	return c.NEmbd
}

// k3Config carries the container's Kimi-K3 fields into c and checks they
// describe a graph the engine runs. nact is how many activation flags the
// container set; situ is the only activation then.
func k3Config(out *Config, c *jlm.Config, nact int) error {
	out.AttnResBlock = int(c.AttnResBlock)
	out.ExpertLatent = int(c.ExpertLatent)
	out.KDALowerBound = c.KDALowerBound
	out.LatentNormEps = float64(c.LatentNormEps)
	if out.LatentNormEps == 0 {
		out.LatentNormEps = out.RMSEps
	}
	if c.SituBeta != 0 || c.SituLinearBeta != 0 {
		if nact != 0 || out.Act != graph.ActSiLU {
			return fmt.Errorf("model: %s: the container claims two activations", out.Arch)
		}
		out.Act = graph.ActSitu
	}
	switch {
	case out.KDALowerBound > 0:
		return fmt.Errorf("model: %s: a decay bound of %g; a bound is negative", out.Arch, out.KDALowerBound)
	case out.KDALowerBound != 0 && !out.ChanDecay():
		return fmt.Errorf("model: %s: a decay bound on a model with no Kimi Delta Attention", out.Arch)
	case out.ExpertLatent != 0 && (out.NExpert == 0 || out.ExpertLatent >= out.NEmbd || out.DenseMoE ||
		out.ExpertWeightIn):
		return fmt.Errorf("model: %s: a latent mixture at %d on %d experts at n_embd %d", out.Arch,
			out.ExpertLatent, out.NExpert, out.NEmbd)
	case out.AttnResBlock < 0 || out.ResAttn() && (out.PredStreams != 0 || out.HCMult != 0 || out.Parallel ||
		out.NMTP != 0):
		return fmt.Errorf("model: %s: residual attention every %d blocks beside another residual "+
			"structure", out.Arch, out.AttnResBlock)
	}
	return nil
}

// loadK3 reads block bi's Kimi-K3 matrices beyond the common ones and checks
// their shapes: the MLA output gate and the latent mixture's projections.
// Its vectors are the block's table's (layer.vecs).
func loadK3(l *Layer, bi int32, cfg *Config, c *jlm.File, getTo func(*Tensor, jlm.Role, int32, int32) error) error {
	var err error
	// The MLA output gate: RoleAttnGate in a full block of an MLA model,
	// NHead*HeadDimV rows over n_embd. (A linear block's RoleAttnGate is its
	// z projection, bound in build.)
	if cfg.MLA() && !cfg.LayerKind(int(bi)).Recurrent() && c.Has(jlm.RoleAttnGate, bi, -1) {
		if err = getTo(&l.MlaGate, jlm.RoleAttnGate, bi, -1); err != nil {
			return err
		}
		if l.MlaGate.K != cfg.NEmbd || l.MlaGate.Rows != cfg.NHead*cfg.HeadDimV {
			return fmt.Errorf("model: block %d's MLA output gate is %dx%d, want %dx%d", bi, l.MlaGate.K,
				l.MlaGate.Rows, cfg.NEmbd, cfg.NHead*cfg.HeadDimV)
		}
	}
	if cfg.ExpertLatent != 0 && cfg.MoEAt(int(bi)) {
		if err = getTo(&l.RoutedDown, jlm.RoleFFNRoutedDown, bi, -1); err != nil {
			return err
		}
		if err = getTo(&l.RoutedUp, jlm.RoleFFNRoutedUp, bi, -1); err != nil {
			return err
		}
		L := cfg.ExpertLatent
		if l.RoutedDown.K != cfg.NEmbd || l.RoutedDown.Rows != L || l.RoutedUp.K != L || l.RoutedUp.Rows != cfg.NEmbd {
			return fmt.Errorf("model: block %d's latent mixture is %dx%d down and %dx%d up; want %d to %d and back",
				bi, l.RoutedDown.K, l.RoutedDown.Rows, l.RoutedUp.K, l.RoutedUp.Rows, cfg.NEmbd, L)
		}
	}
	return nil
}

// loadK3Head reads the head's residual-attention score vector.
func loadK3Head(m *Model, cfg *Config, vec func(jlm.Role, int32) ([]float32, error)) error {
	if !cfg.ResAttn() {
		return nil
	}
	var err error
	if m.ResOut, err = vec(jlm.RoleOutputResScore, jlm.DenseBlock); err != nil {
		return err
	}
	if len(m.ResOut) != cfg.NEmbd {
		return fmt.Errorf("model: the head's residual score is %d wide, want %d", len(m.ResOut), cfg.NEmbd)
	}
	return nil
}

// f32Bytes is a float32 slice addressed as the bytes a kernel reads: the same
// memory, no copy.
func f32Bytes(v []float32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

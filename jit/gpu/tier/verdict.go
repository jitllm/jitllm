package tier

import "github.com/jitllm/jitllm/engine/nn"

// admitPlan is admit for a block's offer, and where a tier that serves by
// verdict (Config.ComposedAuto) takes the composer's parts for the model it
// now holds: only when every device's backend is one the plan's verdict
// covers.
func (g *GPU) admitPlan(p *nn.LayerPlan, what string) (func(), bool) {
	done, ok := g.admit(p.Model, what)
	if !ok || !g.ComposedAuto || g.verdictSet && g.verdictFor == p.Model {
		return done, ok
	}
	if g.verdictAuto {
		g.Composed, g.verdictAuto = 0, false
	}
	if g.Composed == 0 && p.ComposedOn != 0 && g.backends()&^p.ComposedOn == 0 {
		g.Composed, g.verdictAuto = ComposedStep, true
	}
	g.verdictFor, g.verdictSet = p.Model, true
	return done, ok
}

func (g *GPU) backends() nn.Backends {
	var b nn.Backends
	for _, d := range g.devs {
		switch d.dev.API() {
		case "ptx":
			b |= nn.BackendPTX
		case "spirv":
			b |= nn.BackendSPIRV
		case "msl":
			b |= nn.BackendMSL
		default:
			b |= nn.BackendAMDGCN
		}
	}
	return b
}

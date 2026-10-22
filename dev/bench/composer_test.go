package bench

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestComposerAgainstMain measures a decode token of the composed plan
// against main's on one device and one backend, each model's every block
// placed there by main's tier: a soak of both arms, then main against itself
// -- the self-control the ratio is quoted against (RULE 2) -- then main
// against the plan twice, each a paired, interleaved bench.AB.
//
//	JITLLM_COMPOSER_MODELS  comma-separated container names or paths (required)
//	JITLLM_COMPOSER_API     ptx, spirv or msl (default ptx)
//	JITLLM_AB_ROUNDS        rounds of each comparison (default 20)
//	JITLLM_SOAK             seconds both arms run before measuring (default 10)
//
// A ratio above 1 is the plan faster. A row where main places fewer blocks
// than the model has says so and is not main's device rate.
func TestComposerAgainstMain(t *testing.T) {
	names := os.Getenv("JITLLM_COMPOSER_MODELS")
	if names == "" {
		t.Skip("set JITLLM_COMPOSER_MODELS")
	}
	api := os.Getenv("JITLLM_COMPOSER_API")
	if api == "" {
		api = "ptx"
	}
	rounds := envInt("JITLLM_AB_ROUNDS", 20)
	soak := time.Duration(envInt("JITLLM_SOAK", 10)) * time.Second
	const iters = 32
	gpu := jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}
	for _, name := range strings.Split(names, ",") {
		path := testmodels.Resolve(name)
		var dev backend.Device
		for _, d := range backend.Open() {
			if d.API() == api && dev == nil {
				dev = d
				continue
			}
			d.Close()
		}
		if dev == nil {
			t.Fatalf("no %s device", api)
		}
		mainM, err := model.Open(path)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		planM, err := model.Open(path, model.WithPlanDevice(dev, gpu))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		// A capacity the model's context allows; a State that fills is
		// replaced inside the round that fills it.
		capacity := min(4096, mainM.Cfg.NCtx)
		g, err := tier.OpenWith(tier.WithDevices(api))
		if err != nil {
			t.Fatalf("%s: main's tier on %s: %v", name, api, err)
		}
		mainS, planS := mainM.NewState(capacity), planM.NewState(capacity)
		mainS.SetDeviceLayers(g, mainM.Cfg.NLayer)
		// A State's positions run out at its capacity; a fresh one takes
		// over, outside nothing but the round that crosses it.
		step := func(m *model.Model, s **model.State, place bool) func(int) {
			return func(n int) {
				for range n {
					if (*s).Pos() >= capacity-1 {
						(*s).Close()
						*s = m.NewState(capacity)
						if place {
							(*s).SetDeviceLayers(g, m.Cfg.NLayer)
						}
					}
					if _, err := (*s).Forward(int32(1 + (*s).Pos()%100)); err != nil {
						t.Fatalf("%s: %v", name, err)
					}
				}
			}
		}
		mainFn, planFn := step(mainM, &mainS, true), step(planM, &planS, false)
		for end := time.Now().Add(soak); time.Now().Before(end); {
			mainFn(iters)
			planFn(iters)
		}
		placed := 0
		for _, n := range g.Placed() {
			placed += n
		}
		where := fmt.Sprintf("%d of %d blocks on %s", placed, mainM.Cfg.NLayer, dev.Name())
		if placed < mainM.Cfg.NLayer {
			where += fmt.Sprintf(" (main's tier: %v): main's row is not its device rate", g.Err())
		}
		aa := AB(Case{Name: "main", Fn: mainFn}, Case{Name: "main again", Fn: mainFn}, rounds, iters)
		t.Logf("%s, %s: A/A %s", name, where, aa)
		if !aa.Stable() || aa.Median < 0.97 || aa.Median > 1.03 {
			t.Errorf("%s: the self-control reads %.4fx: the harness cannot quote a ratio here", name, aa.Median)
		} else {
			for pass := 1; pass <= 2; pass++ {
				r := AB(Case{Name: "main", Fn: mainFn}, Case{Name: "plan", Fn: planFn}, rounds, iters)
				t.Logf("%s pass %d: %s; main %.1f tok/s, plan %.1f tok/s", name, pass, r,
					float64(iters)/r.AMedian.Seconds(), float64(iters)/r.BMedian.Seconds())
			}
		}
		mainS.Close()
		planS.Close()
		g.Close()
		planM.Close()
		mainM.Close()
		dev.Close()
	}
}

func envInt(name string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(name)); err == nil && v > 0 {
		return v
	}
	return def
}

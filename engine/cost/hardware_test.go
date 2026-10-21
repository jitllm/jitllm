package cost

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
)

// TestHardwareTableIsGenerated holds the compiled-in hardware table to the
// form the renderer writes: entries by name, one per hardware model and
// backend, numbers to four figures. A table edited by hand fails it; the
// fix is the command that measures a machine into it.
func TestHardwareTableIsGenerated(t *testing.T) {
	want, err := RenderHardware(Hardware())
	if err != nil {
		t.Fatal(err)
	}
	have, err := os.ReadFile(filepath.Join("..", "..", HardwareTable))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(have, want) {
		t.Fatalf("%s is not in the form %s writes", HardwareTable, HardwareCommand)
	}
	for _, p := range Hardware() {
		if q, ok := Known(p.Name, p.API); !ok || q != p {
			t.Errorf("%s on %s: Known gives %+v", p.Name, p.API, q)
		}
	}
}

// TestRendererWritesEveryField holds the renderer to writing every field of
// a profile whose every number is set, so a field added to Device reaches
// the table without anyone remembering it.
func TestRendererWritesEveryField(t *testing.T) {
	var p Profile
	var fill func(v reflect.Value)
	fill = func(v reflect.Value) {
		switch v.Kind() {
		case reflect.Struct:
			for i := range v.NumField() {
				fill(v.Field(i))
			}
		case reflect.Array:
			for i := range v.Len() {
				fill(v.Index(i))
			}
		case reflect.Float64:
			v.SetFloat(1.5)
		case reflect.Int, reflect.Int64:
			v.SetInt(3)
		case reflect.Uint8:
			v.SetUint(1)
		case reflect.Bool:
			v.SetBool(true)
		case reflect.String:
			v.SetString("x")
		}
	}
	fill(reflect.ValueOf(&p).Elem())
	b, err := RenderHardware([]Profile{p})
	if err != nil {
		t.Fatal(err)
	}
	var names func(t reflect.Type)
	src := string(b)
	names = func(ty reflect.Type) {
		switch ty.Kind() {
		case reflect.Struct:
			for i := range ty.NumField() {
				f := ty.Field(i)
				if f.Type.Kind() != reflect.Bool && !strings.Contains(src, f.Name+":") {
					t.Errorf("the table leaves out %s.%s", ty.Name(), f.Name)
				}
				names(f.Type)
			}
		case reflect.Array:
			names(ty.Elem())
		}
	}
	names(reflect.TypeFor[Profile]())
	for _, c := range classNames {
		if !strings.Contains(src, c+":") {
			t.Errorf("the table leaves out the class %s", c)
		}
	}
}

// TestEveryKnownHardwarePrices prices the board's fixture with every
// profile in the table, the whole model on that device, as a router would
// before sending a request there: each estimate positive and finite, and
// among the GPUs the step's memory time no longer on twice the bandwidth.
// (Its whole time need not order so: the composed argmax runs on one
// thread, and a thread's speed is not a card's bandwidth.)
func TestEveryKnownHardwarePrices(t *testing.T) {
	f, err := jlm.Open(testmodels.Path(*composeModel + ".jlm"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	s := jit.Step{Rows: 1, Page: 64, Keys: 9}
	type priced struct {
		name string
		bw   float64
		secs float64
	}
	var gpus []priced
	for _, p := range Hardware() {
		c, err := Estimate(jit.Default(), f, s, func(jit.Part) Device { return p.Device })
		if err != nil {
			t.Errorf("%s on %s: %v", p.Name, p.API, err)
			continue
		}
		if !(c.Seconds > 0) || math.IsInf(c.Seconds, 0) || c.Unknown > 0 {
			t.Errorf("%s on %s: %+v", p.Name, p.API, c)
		}
		t.Logf("%-42s %-6s %9.1f us a step, %8.1f us of it memory", p.Name, p.API, c.Seconds*1e6, c.MemSeconds*1e6)
		if p.Device.Kind == jit.DeviceGPU {
			gpus = append(gpus, priced{p.Name, p.Device.Bandwidth, c.MemSeconds})
		}
	}
	for _, a := range gpus {
		for _, b := range gpus {
			if a.bw > 2*b.bw && a.secs > b.secs {
				t.Errorf("%s reads %.0f GB/s and its step's memory takes %.1f us, %s %.0f GB/s and %.1f us", a.name, a.bw/1e9, a.secs*1e6, b.name, b.bw/1e9, b.secs*1e6)
			}
		}
	}
}

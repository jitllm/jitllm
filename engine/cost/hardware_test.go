package cost

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
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

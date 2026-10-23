package hip

import (
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

// TestBogusROCmIsNoDevices: a named ROCm directory that holds nothing is zero
// HIP devices and no error, which is what lets one binary run on every host.
// It is the user's requirement, and the shape a missing ROCm must take so the
// tier goes on to Vulkan.
func TestBogusROCmIsNoDevices(t *testing.T) {
	for _, p := range []string{"/nonexistent/rocm/lib", t.TempDir()} {
		rt, err := Open(Config{Path: p})
		if err != nil {
			t.Errorf("%s: Open returned an error for an absent ROCm: %v", p, err)
		}
		if rt != nil {
			t.Errorf("%s: Open returned a runtime with no ROCm behind it", p)
		}
		// Off Linux and Windows the runtime is refused before any load is tried, and Why
		// says so; on Linux it names the load that failed.
		want := "not loadable"
		if _, ok := layouts[runtime.GOOS]; !ok {
			want = "ROCm runs on Linux and Windows"
		}
		if why := Why(Config{Path: p}); !strings.Contains(why, want) {
			t.Errorf("%s: Why = %q, want it to say %q", p, why, want)
		}
		if _, err := LoadComgr(Config{Path: p}); err == nil {
			t.Errorf("%s: comgr loaded from a directory that has none", p)
		}
	}
}

// TestNamedPathIsTheOnlyOneTried: a caller who names a ROCm gets that one or
// none. A search that fell back to the default places would load a different
// ROCm than the one asked for.
func TestNamedPathIsTheOnlyOneTried(t *testing.T) {
	dir := filepath.Join(string(filepath.Separator)+"x", "lib")
	lo, ok := layouts[runtime.GOOS]
	if !ok {
		// No ROCm layout here (macOS): the search logic is the same on any
		// host, so it is checked against Linux's.
		lo = layouts["linux"]
	}
	got := candidates(Config{Path: dir}, lo, false)
	if len(got) != len(lo.hip) {
		t.Fatalf("candidates = %v", got)
	}
	for _, c := range got {
		if filepath.Dir(c) != dir {
			t.Errorf("candidate %q is outside the named directory", c)
		}
	}
	def := candidates(Config{}, lo, false)
	if len(def) < len(lo.hip) || def[0] != lo.hip[0] {
		t.Errorf("the default search does not start with the bare sonames: %v", def)
	}
}

// TestNewestROCmFirst: /opt/rocm-6.10.0 is newer than /opt/rocm-6.9.2, which a
// string sort gets backwards.
func TestNewestROCmFirst(t *testing.T) {
	cases := []struct{ a, b string }{
		{"/opt/rocm-6.9.2/lib", "/opt/rocm-6.10.0/lib"},
		{"/opt/rocm-6.4.1/lib", "/opt/rocm-7.0.0/lib"},
		{"/opt/rocm-6.4/lib", "/opt/rocm-6.4.1/lib"},
	}
	for _, c := range cases {
		if !versionLess(c.a, c.b) || versionLess(c.b, c.a) {
			t.Errorf("%s should order before %s", c.a, c.b)
		}
	}
}

// TestVulkanUUIDIsRADVs pins the identity to mesa's ac_compute_device_uuid:
// four little-endian u32, domain, bus, device, function. A different byte
// order is a different UUID and leaves one card counted twice.
func TestVulkanUUIDIsRADVs(t *testing.T) {
	d := &Device{pci: [3]int32{0x0001, 0xc3, 0x00}, pciOK: true}
	u, ok := d.VulkanUUID()
	want := [16]byte{1, 0, 0, 0, 0xc3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	if !ok || u != want {
		t.Errorf("VulkanUUID = % x, %v; want % x", u, ok, want)
	}
	d.pciOK = false
	if _, ok := d.VulkanUUID(); ok {
		t.Error("an unknown PCI location produced an identity")
	}
}

// TestArchFromPropsRefusesALayoutItDoesNotKnow: gcnArchName is read at a fixed
// offset, so a runtime with a different struct must be refused, not read.
func TestArchFromPropsRefusesALayoutItDoesNotKnow(t *testing.T) {
	p := make([]byte, propsBuf)
	copy(p[propsArchOffset:], "gfx942:sramecc+:xnack-\x00")
	if a, err := archFromProps(p); err != nil || a != "gfx942:sramecc+:xnack-" {
		t.Errorf("archFromProps = %q, %v", a, err)
	}
	copy(p[propsArchOffset:], "Radeon\x00")
	if _, err := archFromProps(p); err == nil {
		t.Error("a non-gfx name was accepted as an arch")
	}
}

// TestNoExternalCommands: the AMD backend reaches ROCm through its libraries
// only. Neither this package nor the lowering nor any production file of the
// engine's libraries (jit, backend, engine, format, tok, convert) imports os/exec; the
// commands under cmd/, ui/ and scripts/ are programs, not the engine.
func TestNoExternalCommands(t *testing.T) {
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	seen := 0
	for _, top := range []string{"jit", "backend", "engine", "format", "tok", "convert"} {
		err := filepath.WalkDir(filepath.Join(root, top), func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() && d.Name() == "testdata" {
				return filepath.SkipDir
			}
			if d.IsDir() || !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
				return nil
			}
			f, err := parser.ParseFile(token.NewFileSet(), p, nil, parser.ImportsOnly)
			if err != nil {
				return err
			}
			seen++
			for _, im := range f.Imports {
				if path, _ := strconv.Unquote(im.Path.Value); path == "os/exec" {
					rel, _ := filepath.Rel(root, p)
					t.Errorf("%s imports os/exec: the engine runs no external command", rel)
				}
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if seen < 100 {
		t.Fatalf("scanned %d files; the walk is not reaching the tree", seen)
	}
	t.Logf("%d production files scanned", seen)
}

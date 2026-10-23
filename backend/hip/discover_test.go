package hip

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// touch creates the empty files named under root.
func touch(t *testing.T, root string, files ...string) {
	t.Helper()
	for _, f := range files {
		p := filepath.Join(root, f)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// TestWindowsSDKIsFoundNewestFirst lays out two HIP SDKs the way the
// installer does (Program Files\AMD\ROCm\X.Y\bin) and checks the search: the
// bare DLL names first, for the loader's own path (the driver puts
// amdhip64_N.dll in System32), then each SDK newest first by number (6.10
// after 6.4), and in each the comgr whose name carries its version, which no
// fixed list could know.
func TestWindowsSDKIsFoundNewestFirst(t *testing.T) {
	root := t.TempDir()
	touch(t, root,
		"AMD/ROCm/6.4/bin/amdhip64_6.dll",
		"AMD/ROCm/6.4/bin/amd_comgr0604.dll",
		"AMD/ROCm/6.10/bin/amdhip64_6.dll",
		"AMD/ROCm/6.10/bin/amd_comgr0610.dll",
		"AMD/ROCm/6.10/bin/hiprtc0610.dll",
		"AMD/Other/bin/amdhip64_6.dll",
	)
	lo := layouts["windows"]
	lo.root = root
	hipC := candidates(Config{}, lo, false)
	want := append(slices.Clone(lo.hip),
		filepath.Join(root, "AMD/ROCm/6.10/bin/amdhip64_6.dll"),
		filepath.Join(root, "AMD/ROCm/6.4/bin/amdhip64_6.dll"))
	if !slices.Equal(hipC, want) {
		t.Errorf("HIP candidates\n got %v\nwant %v", hipC, want)
	}
	cg := candidates(Config{}, lo, true)
	want = append(slices.Clone(lo.comgr),
		filepath.Join(root, "AMD/ROCm/6.10/bin/amd_comgr0610.dll"),
		filepath.Join(root, "AMD/ROCm/6.4/bin/amd_comgr0604.dll"))
	if !slices.Equal(cg, want) {
		t.Errorf("comgr candidates\n got %v\nwant %v", cg, want)
	}

	// A named directory (HIP_PATH's bin, or JITLLM_ROCM) is the only one
	// tried, and its own comgr is found by pattern.
	named := filepath.Join(root, "AMD/ROCm/6.4/bin")
	cg = candidates(Config{Path: named}, lo, true)
	if !slices.Equal(cg, []string{filepath.Join(named, "amd_comgr0604.dll")}) {
		t.Errorf("named directory's comgr candidates = %v", cg)
	}
	// An empty named directory still names what it looked for.
	empty := t.TempDir()
	cg = candidates(Config{Path: empty}, lo, true)
	if len(cg) != len(lo.comgr) || filepath.Dir(cg[0]) != empty {
		t.Errorf("empty named directory's candidates = %v", cg)
	}
}

// TestLinuxLayoutUnderARoot: the same search on Linux, against a fake root, so
// the two layouts are held to one shape.
func TestLinuxLayoutUnderARoot(t *testing.T) {
	root := t.TempDir()
	touch(t, root,
		"opt/rocm-6.4.1/lib/libamdhip64.so.6",
		"opt/rocm-6.4.1/lib/libamd_comgr.so.3",
		"opt/rocm-7.0.0/lib/libamdhip64.so.7",
	)
	lo := layouts["linux"]
	lo.root = root
	got := candidates(Config{}, lo, false)
	want := append(slices.Clone(lo.hip),
		filepath.Join(root, "opt/rocm-7.0.0/lib/libamdhip64.so.7"),
		filepath.Join(root, "opt/rocm-6.4.1/lib/libamdhip64.so.6"))
	if !slices.Equal(got, want) {
		t.Errorf("HIP candidates\n got %v\nwant %v", got, want)
	}
}

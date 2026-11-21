package backend_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// validationLayer is the Khronos validation layer's name, and validationManifest
// the file the loader finds it by.
const (
	validationLayer    = "VK_LAYER_KHRONOS_validation"
	validationManifest = "VkLayer_khronos_validation.json"
)

// TestSplitLaunchesAreValidVulkanUsage reruns the two split gates in a child
// process under the Khronos validation layer and fails on any validation error.
//
// It is the half the data gates cannot supply. A dispatch past
// maxComputeWorkGroupCount is invalid usage, and an integrated GPU's driver
// can run one correctly anyway -- an unsplit launch of 1048576 workgroups wrote
// every word right -- so a buffer check passes the old behaviour on a device
// with a small limit. The layer does not: it reports
// VUID-vkCmdDispatch-groupCountX-00386 for every such dispatch, whatever a given
// driver makes of it.
//
// JITLLM_VK_VALIDATION is the prefix the layer is installed under (default
// /usr): its share/vulkan/explicit_layer.d holds the manifest and its
// lib/<triple> the library. A package nobody installed can be unpacked without
// root: apt-get download vulkan-validationlayers && dpkg-deb -x *.deb DIR, then
// JITLLM_VK_VALIDATION=DIR/usr.
func TestSplitLaunchesAreValidVulkanUsage(t *testing.T) {
	prefix := os.Getenv("JITLLM_VK_VALIDATION")
	if prefix == "" {
		prefix = "/usr"
	}
	manifests := filepath.Join(prefix, "share", "vulkan", "explicit_layer.d")
	if _, err := os.Stat(filepath.Join(manifests, validationManifest)); err != nil {
		t.Skipf("VALIDATION LAYER MISSING: %v (apt-get download vulkan-validationlayers && dpkg-deb -x "+
			"the .deb into DIR, then JITLLM_VK_VALIDATION=DIR/usr) -- this gate proved nothing", err)
	}
	libs, _ := filepath.Glob(filepath.Join(prefix, "lib", "*-linux-gnu"))
	libs = append(libs, filepath.Join(prefix, "lib"))
	if old := os.Getenv("LD_LIBRARY_PATH"); old != "" {
		libs = append(libs, old)
	}

	cmd := exec.Command(os.Args[0], "-test.run",
		"^(TestALaunchPastTheGroupLimitRunsWhole|TestRealKernelsPastTheGroupLimit)$",
		"-test.count=1", "-test.v")
	cmd.Env = append(os.Environ(),
		"VK_LAYER_PATH="+manifests,
		"VK_INSTANCE_LAYERS="+validationLayer,
		"VK_LOADER_DEBUG=layer",
		"LD_LIBRARY_PATH="+strings.Join(libs, ":"))
	out, err := cmd.CombinedOutput()
	text := string(out)
	if vuids := regexp.MustCompile(`VUID-[A-Za-z0-9_-]+`).FindAllString(text, -1); len(vuids) > 0 {
		seen := map[string]int{}
		for _, v := range vuids {
			seen[v]++
		}
		t.Fatalf("the validation layer reported %d error(s) (child: %v): %v", len(vuids), err, seen)
	}
	if err != nil {
		t.Fatalf("the split gates failed under the validation layer: %v\n%s", err, tail(text, 4000))
	}
	// The selection check: a layer that did not load reports nothing, which
	// reads exactly like a clean run.
	if !strings.Contains(text, `Inserted device layer "`+validationLayer+`"`) {
		t.Fatalf("the loader never inserted %s into a device, so nothing was validated\n%s",
			validationLayer, tail(text, 4000))
	}
	if !strings.Contains(text, "--- PASS: TestALaunchPastTheGroupLimitRunsWhole") ||
		!strings.Contains(text, "--- PASS: TestRealKernelsPastTheGroupLimit") {
		t.Fatalf("the split gates did not both run and pass in the child\n%s", tail(text, 4000))
	}
}

func tail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return "..." + s[len(s)-n:]
}

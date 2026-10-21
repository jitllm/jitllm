package backend

import (
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/vulkan"
)

// softwareChild marks the child process TestSoftwareVulkanIsDeclined runs its
// body in.
const softwareChild = "JITLLM_VK_SOFTWARE_CHILD"

// TestSoftwareVulkanIsDeclined drives openVulkan on a loader that can only see
// Mesa's lavapipe, and demands an error. llvmpipe answers every call
// correctly, so nothing downstream would catch blocks placed on it; this is the
// only place, before the Device reaches the list. VK_ICD_FILENAMES restricts
// the loader to one driver.
//
// The body runs in a child process with the restriction in its environment
// from the start: the loader reads it when the process's one VkInstance is
// created, which in this process an earlier test may already have done.
func TestSoftwareVulkanIsDeclined(t *testing.T) {
	const lvp = "/usr/share/vulkan/icd.d/lvp_icd.json"
	if _, err := os.Stat(lvp); err != nil {
		t.Skipf("lavapipe ICD not installed: %v", err)
	}
	if os.Getenv(softwareChild) == "" {
		cmd := exec.Command(os.Args[0], "-test.run", "^TestSoftwareVulkanIsDeclined$", "-test.count=1", "-test.v")
		cmd.Env = append(os.Environ(), softwareChild+"=1", "VK_ICD_FILENAMES="+lvp, "VK_DRIVER_FILES="+lvp)
		out, err := cmd.CombinedOutput()
		text := string(out)
		switch {
		case strings.Contains(text, "--- SKIP: TestSoftwareVulkanIsDeclined"):
			t.Skipf("the child skipped:\n%s", text)
		case err != nil || !strings.Contains(text, "--- PASS: TestSoftwareVulkanIsDeclined"):
			t.Fatalf("the child did not pass (%v):\n%s", err, text)
		}
		t.Logf("child:\n%s", text)
		return
	}
	// No pinned device: the default rule must decline a software-only loader.
	d, err := openVulkan(Opts{})
	if err == nil {
		name := d.Name()
		d.Close()
		t.Fatalf("openVulkan returned %q on a software-only loader; it must decline", name)
	}
	if !strings.Contains(err.Error(), "declining") {
		t.Skipf("no lavapipe device to decline here: %v", err)
	}
	t.Logf("declined: %v", err)

	// Naming it is an explicit request, and must work: that is what keeps a
	// software-only CI host able to run the SPIR-V tests.
	d, err = openVulkan(Opts{Vulkan: vulkan.Config{Device: "0"}})
	if err != nil {
		t.Fatalf("a pinned device 0 must take the software device: %v", err)
	}
	defer d.Close()
	if !strings.Contains(d.Name(), "SOFTWARE RASTERISER") {
		t.Errorf("the software device opened as %q, with no warning in the name", d.Name())
	}
	free, total, merr := d.Mem()
	t.Logf("%s: %.2f of %.2f GiB free, err=%v, unified=%v", d.Name(),
		float64(free)/(1<<30), float64(total)/(1<<30), merr,
		d.(Unified).UnifiedMemory())
}

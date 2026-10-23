package tier

import (
	"reflect"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/backend"
)

// countKern is a backend.Kernel that records its own Close; the package's
// other fakes discard it, so no gate could see a kernel handle leak.
type countKern struct{ closed *int }

func (k *countKern) Close()                                              { *k.closed++ }
func (k *countKern) Launch(groups, width int, bufs ...backend.Buf) error { return nil }

// TestFreeScratchClosesEveryKernel fills every backend.Kernel field of a
// blockScratch and requires freeScratch, which runs on every KV grow, to close
// all of them. The fields are found by reflection, so a field added later is
// covered without editing this test.
func TestFreeScratchClosesEveryKernel(t *testing.T) {
	closed := 0
	bs := &blockScratch{}
	v := reflect.ValueOf(bs).Elem()
	kernType := reflect.TypeOf((*backend.Kernel)(nil)).Elem()

	want := 0
	for i := 0; i < v.NumField(); i++ {
		f := v.Field(i)
		if f.Type() != kernType {
			continue
		}
		// Unexported, so the value has to be re-derived at its address --
		// the same reason freeScratch itself uses NewAt.
		reflect.NewAt(f.Type(), unsafe.Pointer(f.UnsafeAddr())).
			Elem().Set(reflect.ValueOf(&countKern{closed: &closed}))
		want++
	}
	if want < 10 {
		t.Fatalf("only %d backend.Kernel fields found on blockScratch -- this gate "+
			"proved nothing, and the walk it is checking would too", want)
	}

	freeScratch(bs)

	if closed != want {
		t.Errorf("freeScratch closed %d of %d kernels; the rest leak, and it runs on "+
			"every KV grow", closed, want)
	}
}

// TestFreeScratchLeavesCachedKernelsOpen: scratch kernels are owned by the
// scratch, but matvec kernels come from g.kerns, a device-wide cache keyed by
// shape and shared by every scratch. They reach the scratch inside mv, which
// freeScratch's type filter skips; recursing into struct fields would close a
// kernel the cache still hands out.
func TestFreeScratchLeavesCachedKernelsOpen(t *testing.T) {
	closed := 0
	shared := &countKern{closed: &closed}
	bs := &blockScratch{mvHead: mv{kern: shared, red: shared}}

	freeScratch(bs)

	if closed != 0 {
		t.Errorf("freeScratch closed a kernel reached through mv %d time(s). Those come "+
			"from g.kerns, which is keyed by SHAPE and shared across every scratch on "+
			"the device -- closing one here leaves the cache handing a closed kernel to "+
			"the next block of that shape", closed)
	}
}

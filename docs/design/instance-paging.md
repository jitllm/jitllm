# Paging is the instance's

Stated by the user: a model has several instances, each paging -- if it pages
-- a different part of the model. AGENTS.md's shape says the same of an
instance: "every kernel for one placement (which device each block runs on,
**how heavily it pages**)". This file is the design; the rule is AGENTS.md's.

## What was wrong, before step 1

The container was the file and the pager in one (`jlm.File`): one page array
(`pages`), one budget (`SetBudget`), one resident set, its free lists, the
dense region and the expert pages' reservation. `Model.SetPageBudget` sets that
one budget (and re-binds every block); the server and `cmd/jitllm` set it once
per model. The model's block weights are bound into its frames (`bindPacked`
re-points `model.Layer`'s tensors at a page-in), and `Model.CpuUse` counts the
instances running a block on the CPU so one does not unbind what another runs
(`Model.Bind`).

So:

- an instance has no budget and no resident set of its own: one instance's
  page-in evicts a block another is about to run, and MRU over the cyclic
  scan (`sched/cyclic.go`) is computed over two scans interleaved;
- the model holds the engine's devices, because its pager is the only pager:
  the cards told of a recycled frame (`HeldBy`, `ForgetCopies`), the devices
  released at its Close (`nn.ModelOwner`), the host-memory devices netted off
  its budget (`HoldOnHost`).

## The shape

**The model** is the weights as a file: the container's header, its page
table (every block's, expert's and vision block's offset and size), the
config, the graphs, and the layer descriptors (roles, shapes, where a tensor
lies inside its page). It reads a page or a page's ranges into memory it is
handed (O_DIRECT, as now). It holds no frame -- the dense region included --
no budget, no resident set, no device and no callback.

**Every device pages the blocks placed on it, the same way.** The CPU and a
card are alike: a device reads the pages of the blocks an instance places on
it from the file into memory of its own, keeps them under that instance's
quota on it, evicts in that instance's order, and holds a block two
instances place on it once. One contract, every device implementing it; no
device pages through another's memory, and there is no ledger beside the
devices. The File only reads into memory it is handed.

**An instance's budget is a retention quota.** It bounds the bytes that
instance keeps pinned between uses, shared or not, and its retention order is
MRU over its own scan (LRU for expert pages) -- an instance's policy, not a
claim of global optimality over a shared cache. Unpinned frames stay as the
ledger's clean cache under its own deterministic order; physical pressure
reclaims those first, then asks instances to relinquish, and never recycles a
pinned frame. Admission is at placement, as a KV reservation is: an instance
whose working set (the largest run of blocks it must hold at once) does not
fit what the ledger can give waits or is refused there, never mid-token.

**One lifetime protocol for a frame**: acquire (a hit pins; a miss reserves a
frame, the reads fill it, publishing makes it readable), leases while a block
executes or transfers (a lease outlives an unpin), and a generation that
moves when a frame is reused -- a card's copy keyed on a host address keys on
the generation too, so a reused frame invalidates it without a callback
registered anywhere. A device's hold of a model's blocks is reference counted
by the instances placing them: the last instance's release gives them back,
never one instance's Close while another still runs them.

**An instance's weights point into the frames it leased**: its own bindings
(the CPU form's roles, the `nn.LayerWeights` it offers a card), not
`model.Layer`'s.

The rules that stand: no mmap of files (frames are anonymous memory, pages read
with O_DIRECT); every expert its own page; paging is always on (a model that
fits is an instance whose quota holds every page it runs); a warm token and a
warm fault allocate nothing (the ledger's bookkeeping is preallocated).

## Steps, each its own commit and gates

1. **One pager and binding object the engine goes through.** Every page-in,
   ensure, bind, budget and recycle the engine asks of the model goes through
   one engine object over today's model-wide pager, with today's behaviour
   kept: `Model.PageIn`, `PageInSelected`, `EnsureBlock*`, `SetPageBudget`,
   `ForgetCopies`, `HeldBy`, `HoldOnHost` lose their engine callers. Gates:
   the paging principle's (`TestPagingDoesNotChangeTheAnswer`,
   `TestOpenHonoursAPageBudget`, `TestForwardBatchFaultsItsBlocksIn`,
   `TestTwoPagedModels*`, `TestWindowedLayersHoldTheirWindow`,
   `TestBlocksPageOnTheListedCard`), the expert paging gates,
   `TestDecodeDoesNotAllocate`.
2. **Bindings to the instance.** `model.Layer` loses its addresses; an
   instance binds from the frames it leased. `Model.CpuUse`, `Model.Bind` and
   the re-bind in `SetPageBudget` go. Gates: as 1, plus the relocation
   principle's (a block moving off and back re-binds).
3. **Frames out of the file**, in two commits.
   - **3a**: `jlm.Frames` is split from `jlm.File`. The File keeps the
     header, the table, the layout (page bounds, extents) and the reads into
     memory it is handed; Frames hold the frames, the dense region, the
     budget and resident set, the expert reservation, the read service,
     preload and speculation, over a File (`jlm.NewFrames`). The model still
     opens both (`Model.Frames`); every caller of a frame method is
     re-pointed at the frames.
   - **3b**: each device pages for itself, through one contract every
     device implements: the CPU's into its memory, a card's into its own,
     each reading the file directly -- a card stops reading through the
     CPU's frames. The model opens the File alone and reads its vectors from
     it at open; an instance binds its own copies of its blocks and dense
     tensors at the pages the device running them holds. Address-keyed
     copies key on the page's generation; `OnRecycle` goes. A device's
     frames take the memory it gives them (`jlm.WithFrameMemory`: a card's
     page-locked staging where its backend has it, `backend.HostPinner`). A
     card is handed a block's frames before admission (`LayerWeights.Bind`:
     claimed, unread, so admission reads shapes and lengths as it did) and
     again to fill them once it admits (`Ensure`); a card that declines lets
     its claim go, and an uploaded block's staging page is dropped. (A shared
     engine-level ledger, and the same pager filed under the CPU device,
     were both taken out: each was the model-wide pager under another
     owner, and the second treated the CPU unlike a card.)

     The CPU's half: the CPU device holds each model's frames
     (`cpudev.CPU.Frames`), made by `model.Model.NewFrames` as the caller
     asked at Open (placement, fill granularity, heap, preload); the
     model opens the File alone, reads its vectors from it, and keeps its
     tensors as descriptors. An instance binds its own copy of the dense
     tensors (`model.Dense`: embedding, head, tables, the tower's entry and
     taps) at its CPU device's frames. An instance with no `WithCPU` opens a
     CPU device of its own, and so frames of its own: instances meant to
     share a block's pages share a CPU device, as the server's do. Reads are
     counted on the File across every device's frames (`PagedBytes`,
     `PageFaults`); a front end reads a device's frames through the instance
     (`Instance.FramesOn`).

     A card's frames keep two pages of staging and, beyond it, the banks of
     the blocks it streams (their experts are read every token), capped at
     the instance's page quota (`nn.LayerPlan.FrameBudget` at placement,
     `nn.FrameBudgeter` when the quota changes); the card reports what that
     keeps of the host's memory (`HostReserved`), so the CPU device's frames
     give it up and the two keep what one quota allows.

     A card offered the head reads its tensors -- the projection, the
     token and position tables, the streams' and the collapse's -- from the
     file into its own frames (`nn.Head.File`, `nn.RowsPart.File`) and lets
     that staging go once the upload has copied it (`jlm.Frames.DropDense`),
     rather than uploading from the CPU device's dense copy
     (`engine.TestACardReadsTheHeadIntoItsOwnFrames` poisons the CPU's copy).

     Left of 3b: the CPU device still reads the whole dense region when its
     frames are made, so its copy of a head a card runs is held for nothing
     until the CPU reads dense entries on demand as a card does.
   Gates: as 2, plus the vision and expert gates.
4. **Quotas per instance.** `engine.WithPageBudget`; `Model.SetPageBudget`
   goes; the server's per-model budget becomes each instance's, the
   `SetPageBudget` RPC names the instance (a proto change, buf generate);
   admission at placement. Gates: as 3, and the server's.
   - Done: an instance has a page quota (`engine.WithPageBudget`,
     `Instance.SetPageBudget`; the model's memory plan by default), recorded
     on its CPU device before the frames are made (`cpudev.Quota`) so they
     are budgeted from their first read, a preload's among them
     (`jlm.Budget`). The model holds no budget and no host-memory device:
     `Model.SetPageBudget`, `WithPageBudget`, `PageAsk` and the registry
     (`HoldOnHost`) are gone; what the attached device holds of the host's
     memory comes off its own instance's quota. The server hands its
     per-model share to the model's instance; the CLI and the speed tool
     hand theirs to theirs. A statistic reads the frames a device holds and
     makes none (`cpudev.CPU.HeldFrames`).
   - The CPU device's frames keep what the largest net quota of the
     instances holding the model allows, not the sum: they hold a block every
     instance runs once, and two instances running the same blocks under
     quotas of X keep X. Each instance keeping its own order within its own
     quota over blocks it alone runs -- the retention quota above -- is not
     built: the frames have one order.
   - The RPC still names the model: the server makes one instance a model,
     and names it by its model; a second instance of a model in the server
     is what an instance id in the request waits for.
   - Admission at placement admits every quota: one below the block's
     working set holds one page at a time (a page-in always fits one page,
     `makeRoom`), so an instance is never refused for its quota and never
     fails mid-token for it; it thrashes, which the paging gates run at a
     one-byte quota on purpose. Refusing it would be a refusal where the
     behaviour is implemented.
   - Left: per-instance retention within shared frames -- what this file's
     first "what was wrong" bullet names, still true of two instances on one
     CPU device: one order, one budget. Where it hooks, as read: every page
     use goes through `Frames.claimFor` (the order: `tick`, `recent`; the
     room: `makeRoom`), so a retainer -- an instance's view of the frames,
     its quota, its held bytes, its own `recent` and expert ticks, a bit in a
     per-page retention mask -- is carried into it by every claim:
     `EnsurePage`, `EnsureRanges`, `Claim`, `Hold`, `EnsureExperts` and the
     speculation queue's requests, whose workers claim on another goroutine.
     A claim retains the page for its retainer; room is made by unretaining
     that retainer's own victim (its expert LRU, else its MRU block), and a
     page leaves when no retainer holds it and nothing pins it. `Keep` and
     `DropPage` become retention (an instance releasing a block it placed
     unretains its page; another instance's retention keeps it). The model's
     frame-taking functions (33 of them) take a pager interface both
     `*jlm.Frames` (a card's, frames-level) and a retainer satisfy. The free
     lists' bound becomes the quotas' sum; a preload reads only while every
     quota holds every page. The gate is two instances on one CPU device
     with quotas below their scans, each reading what it alone reads.
5. **The model holds nothing of a device.** Its registries go; `srcgate` holds
   that no type of `model/` has a field of an `nn` device interface, and that
   `Model.Close` reaches no device.
   - Done: a device counts the instances holding a model on it
     (`nn.ModelOwner.HoldModel`); an instance holds its CPU device and the
     device it attaches, the last release gives back what the model placed,
     and `Model.Close` reaches no device
     (`engine.TestADeviceKeepsAModelForItsInstances`).
   - Done: the forgetters. A card keys a block's copies on its block and
     slot, and a tensor at no block's slot (the head, an input table, a
     picture's rows) on its container entry (`nn.Weight.Entry`), not on the
     address of whichever device's memory its bytes were handed in: two
     instances' copies are one, and no later mapping of that memory aliases
     a stale key. Nothing on a card is then keyed on CPU memory that is
     reused, so `Model.ForgetCopies`, `HeldBy`, the CPU frames' recycle
     callback, `nn.CopyForgetter` and `GPU.Forget` go; a card's own frames
     still tell that card of a frame they reuse. The lock serialising the
     readers of evicting frames is the frames' (`jlm.Frames.Enter`), not the
     model's: it protects the reuse of their memory, and instances on two
     CPU devices never wait on each other. `srcgate.TestTheModelHoldsNoDevice`
     holds that no `model/` type has a field naming an `nn` interface and that
     `Model.Close` calls nothing but its container's Close. (The host-memory
     devices netted off the model's budget went with step 4's quotas.)
   - The model's func fields are a clock (`WithChatClock`), an expert
     sheet's release and a test's restatement of the stored tensors, none a
     device's; the check is on `nn`'s interfaces, not on funcs.
6. **The transitions, gated.** Two instances of one model -- one at half its
   pages on the CPU, one with a seam on a card -- each matching the
   reference, the blocks both run held once (the ledger's frame count), with
   overlapping scans; the last owner closing while the other runs; a quota
   shrinking during a read; expert, vision and dense pages covered; a warm
   fault and re-bind allocating nothing; cards-only paging; every leasing
   session's history relocating with its blocks. The violation is the
   model-wide pager (step 1's tree).
   - Done: `TestAnInstanceOnTheCPUAndOneOnACardShareTheFrames` (one instance
     on the CPU under half its pages, one with half its blocks on a card,
     decoding at once from the frames they share, each as alone, then the
     first closing while the other decodes on), `TestTwoInstancesReadAPageOnce`
     (two instances on one CPU device read every page once between them;
     two CPU devices reading every page twice is its violation, seen) and
     `TestAQuotaShrinksUnderARunningDecode` (quotas falling to a page and
     back under two running decodes, each as alone). Writing the last showed,
     from the code, that a budget change could evict a page a CPU block was
     still reading and give its frame back: the CPU's block runs pin nothing,
     and the frames' lock was taken only while they evicted. Every reader of the frames now comes in
     through `jlm.Frames.Enter` -- shared while they fit, alone while they
     evict -- and while one is in, `SetBudget` records the budget and a
     shared reader's page-in evicts nothing; the last reader to leave fits
     the frames to it. The quota is an atomic: it may be set from another
     goroutine than the instance's.
   - Already gated before this step: the last owner closing
     (`TestADeviceKeepsAModelForItsInstances`), a quota reaching the frames
     before their first read (`TestAQuotaBudgetsTheFramesFromTheirFirstRead`),
     a warm fault allocating nothing (`TestPagedDecodeDoesNotAllocate`),
     cards-only paging (`TestBlocksPageOnTheListedCard`), sessions' history
     relocating with their blocks (the relocation principle's gates).
   - The CPU device's frames page in only inside an entry (an instance's
     block runs and expert fills); a placement giving a block's page back
     (`ReleaseLayer`) cannot take one another instance runs from host
     memory, which that instance keeps (`Kept`) -- the frames refuse a kept
     or pinned page.
   - A tower's pages under two instances (`TestTwoInstancesReadATowersPagesOnce`)
     and a mixture's experts beside a card's placement
     (`TestAnInstancesBlocksAreItsOwn` counts the CPU instance's page-ins
     against one alone's) are gated too. What is left of the design is
     step 4's: each instance keeping its own order within its own quota over
     frames it shares.

## Open

- Admission's two modes: a placement that guarantees independent progress
  reserves its working set; one that relies on sharing reserves only what is
  not already pinned, and says so.
- `WithPreload` reads every page when a device's frames are made (the first
  instance holding the model there), under the quota that made them: it
  becomes a warm of the clean cache, nothing pinned until an instance pins it.

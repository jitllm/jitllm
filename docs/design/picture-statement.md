# A picture's preprocessing, stated

Each picture step kind (`jlm.PreKind`) past the generic ones is one
processor's resize: `PreSmartResize` is Qwen2-VL's smart_resize, `PreNaFlex`
SigLIP2's, `PreAspectBudget` Gemma 4's, `PreCapUnit` Pixtral's, `PreLongBox`
Janus's, `PreLimitPad` Kimi-VL's, `PreTile` InternVL's grid, `PreBestFit`
Llama 4's canvas, `PreTileSmart` LFM2-VL's, `PreSliceUHD` MiniCPM-V's. Each
has a branch in `model.PictureFit`, `TowerGraphs.Square`, `Tiles`,
`Window` and `classSides`. A step kind with one family behind it is that
family's code (AGENTS.md, "the architecture is the converter's alone"), as
the video record's flags were (docs/design/video-statement.md). This is the
statement that replaces them: the converter writes each processor's
geometry, the engine evaluates it.

## What is already generic

The pixels are: a resize (Pillow's or torchvision's taps, or two taps, in
the stated arithmetic) to a size fixed at conversion or given by the call
(`graph.Attrs.PicLive`), the result laid on a canvas of a fill colour
(`PicCanvas`, `PicFill`) -- a window of it, and the normalisation
(`PreNormalize`, `PreNormalizeBF16`). The picture op
(`composer/decompose_image.go`) reads the sizes and the canvas from the
call's request words.

What is per family is the geometry: the size a picture resizes to, how many
squares it becomes, each square's window, grid and marker rows, and the
largest area a picture reaches (which sizes the classes formed at
placement).

## The statement

    Geometry    bindings, in order: each a value (an expression, jlm.Expr,
                of the picture's w and h and the bindings before it), or a
                choice from a table the converter writes
    Count       the squares the picture becomes: an expression
    Square      square i's resize size (Ow, Oh), window (X0, Y0, Pw, Ph),
                patch grid (Gw, Gh), canvas (Cw, Ch) and corner on it, and
                its marker runs (Mark, MarkN, Trail, TrailN): an expression
                each, of w, h, i, the count and the bindings
    Bounds      the largest area and patch count a square reaches, and the
                unit its sides are whole multiples of: what the classes
                formed at placement hold
    Steps       resize (filter, arithmetic), place (fill colour),
                normalise -- the pixels' generic steps

A choice scans the table's rows in order: the first eligible row, then any
row of a lexicographically smaller cost, or of an equal cost where its tie
expression is positive. Its row's columns are bindings. It states the
searches: InternVL's and LFM2-VL's closest grid (cost |w/h - c/r|, the tie
to a grid whose area half-covers the picture), Llama 4's canvas (its
limiting scale, then area), MiniCPM-V's slices (|log aspect| over the
counts either side of the multiple).

NaFlex's bisection is a fixed 24 halvings of [1e-6, 100] at a 1e-5
tolerance: the converter unrolls them into bindings, the midpoint, the
sides it gives and the bound each keeps. No step runs a loop the container
does not state.

The expressions are evaluated on the host in float64, as Python computes
them, which needs constants exact in float64 (`jlm.Arg` holds a float32
today): an exact constant kind, valid only in a host-evaluated expression.

## The engine

`model.PictureSquares` evaluates the statement: a picture bound once, each
square after, its scratch kept by the caller so a warm encode allocates
nothing (the instance's tower run holds one). A square's class is the
smallest that holds its grid. The classes are formed from `Bounds`: a Unit
of 0 says every square is the entry's own size, one class; otherwise class
sides double from 16 units to the area the resize reaches. A video whose
frames are read as pictures under steps of their own states their geometry
beside the picture's (`jlm.Video.Frames`).

The picture op reads the canvas and the picture's corner on it from the
request (`composer.RequestPicCW` to `RequestPicCY`), as it reads the resize
size, and the fill colour from its attributes: one `graph.Attrs.PicCanvas`
lays a live picture on a canvas, a pad and a letterbox alike.

## Migration

1. The statement in `format/jlm` (types, record, validation) and its
   evaluator in `model`, held to round trips, refusals, and the numbers on
   a statement no family has.
2. The converter writes each family's geometry beside its old step, and a
   gate holds the evaluator's squares to the old `Square`'s over every
   fixture and a sweep of sizes (aspects past each threshold, each tie, the
   bounds) -- the old code is held to the processors already, so this holds
   the statement to them.
3. `Square` evaluates the statement and the fits go. With them goes the
   gate of step 2: `engine.TestPictureGeometryHoldsItsSweep` holds each
   tower's squares over the same sweep to the sums the fits gave
   (`engine/testdata/picturegeom.sums`, recorded while step 2's gate held
   the two equal). The picture op then takes the canvas from the call.
4. The family kinds go from `format/jlm`; the record's revision moves and
   an older container is reconverted.

Each step keeps every tower's transformers gates, its card and seam gates,
`TestATallPictureRunsInBands` and the warm allocation gate green.

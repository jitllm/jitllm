package game

import (
	"math"
	"strconv"
)

// The request a decision model is asked is TypeSafe's System One body,
// the one jevman sends: a state (the board drawn as text plus the actors'
// positions) and one choice question per open junction, its options the legal
// directions, each described by the facts the game computed for it. The field
// names and the shape are jevman's, so a model sees the same request in both;
// the wording of the questions is this example's own.

var letters = [NActors]byte{'P', 'B', 'K', 'I', 'C'}

// Legend explains the board's characters to the model.
const Legend = "# wall, - ghost-house door, . pellet, o power pellet, F fruit, P Pac-Man, " +
	"B Blinky, K Pinky, I Inky, C Clyde (lowercase: frightened, can be eaten), e eyes of an eaten ghost"

var personas = [NActors]string{
	"",
	"You are Blinky, the red ghost, and you chase: close in on Pac-Man by the shortest way.",
	"You are Pinky, the pink ghost, and you ambush: aim about four tiles in front of Pac-Man to cut him off.",
	"You are Inky, the cyan ghost, and you flank: come at Pac-Man from the side opposite Blinky so he is caught between you.",
	"You are Clyde, the orange ghost: you chase Pac-Man from afar, but within eight tiles you lose your nerve and go back to your corner (bottom left).",
}

// Request writes the decision request for the questions ps into dst and
// returns it. dst is reused, so a warm game builds a request without growing
// it.
func (g *Game) Request(dst []byte, ps []Point) []byte {
	dst = append(dst, `{"state":`...)
	dst = g.State(dst)
	dst = append(dst, `,"questions":{`...)
	for i := range ps {
		if i > 0 {
			dst = append(dst, ',')
		}
		dst = g.Question(dst, &ps[i])
	}
	return append(dst, "}}"...)
}

// State writes the board as the model sees it.
func (g *Game) State(dst []byte) []byte {
	m := g.Maze
	var grid [64][64]byte
	for y, r := range m.Rows {
		for x := 0; x < m.W; x++ {
			c, k := r[x], y*m.W+x
			switch c {
			case '.':
				c = ' '
				if m.Pellet[k] {
					c = '.'
				}
			case 'o':
				c = ' '
				if m.Power[k] {
					c = 'o'
				}
			case '_':
				c = '#'
			}
			grid[y][x] = c
		}
	}
	put := func(t Tile, c byte) {
		if t.Y >= 0 && t.Y < m.H && t.X >= 0 && t.X < m.W {
			grid[t.Y][t.X] = c
		}
	}
	if g.Fruit.On {
		put(g.Fruit.Tile, 'F')
	}
	for id := Blinky; id < NActors; id++ {
		a := &g.Actors[id]
		switch a.State {
		case Normal:
			put(g.Occupied(a), letters[id])
		case Frightened:
			put(g.Occupied(a), letters[id]+'a'-'A')
		case Eaten:
			put(g.Occupied(a), 'e')
		}
	}
	pac := g.Occupied(&g.Actors[Pacman])
	put(pac, 'P')
	dst = append(dst, `{"maze":[`...)
	for y := 0; y < m.H; y++ {
		if y > 0 {
			dst = append(dst, ',')
		}
		dst = append(dst, '"')
		dst = append(dst, grid[y][:m.W]...)
		dst = append(dst, '"')
	}
	dst = append(dst, `],"legend":`...)
	dst = appendString(dst, Legend)
	dst = append(dst, `,"mode":"`...)
	dst = append(dst, g.Mode.String()...)
	dst = append(dst, `","frightened_seconds_left":`...)
	dst = appendTenth(dst, g.FrightLeft)
	dst = append(dst, `,"pacman":`...)
	dst = appendPos(dst, pac, g.Actors[Pacman].Dir)
	dst = append(dst, '}')
	dst = append(dst, `,"ghosts":{`...)
	for id := Blinky; id < NActors; id++ {
		if id > Blinky {
			dst = append(dst, ',')
		}
		a := &g.Actors[id]
		dst = append(dst, '"')
		dst = append(dst, IDs[id]...)
		dst = append(dst, `":`...)
		dst = appendPos(dst, g.Occupied(a), a.Dir)
		dst = append(dst, `,"state":"`...)
		dst = append(dst, a.State.String()...)
		dst = append(dst, `"}`...)
	}
	dst = append(dst, `},"pellets_left":`...)
	dst = strconv.AppendInt(dst, int64(m.Left), 10)
	dst = append(dst, `,"lives":`...)
	dst = strconv.AppendInt(dst, int64(g.Lives), 10)
	dst = append(dst, `,"level":`...)
	dst = strconv.AppendInt(dst, int64(g.Level), 10)
	dst = append(dst, `,"score":`...)
	dst = strconv.AppendInt(dst, int64(g.Score), 10)
	dst = append(dst, `,"fruit":`...)
	if !g.Fruit.On {
		dst = append(dst, "null"...)
	} else {
		dst = append(dst, `{"kind":"`...)
		dst = append(dst, g.Fruit.Kind...)
		dst = append(dst, `","points":`...)
		dst = strconv.AppendInt(dst, int64(g.Fruit.Points), 10)
		dst = append(dst, `,"x":`...)
		dst = strconv.AppendInt(dst, int64(g.Fruit.Tile.X), 10)
		dst = append(dst, `,"y":`...)
		dst = strconv.AppendInt(dst, int64(g.Fruit.Tile.Y), 10)
		dst = append(dst, `,"seconds_left":`...)
		dst = appendTenth(dst, g.Fruit.Left)
		dst = append(dst, '}')
	}
	return append(dst, '}')
}

// appendPos writes {"x":..,"y":..,"dir":".." without the closing brace.
func appendPos(dst []byte, t Tile, d Dir) []byte {
	dst = append(dst, `{"x":`...)
	dst = strconv.AppendInt(dst, int64(t.X), 10)
	dst = append(dst, `,"y":`...)
	dst = strconv.AppendInt(dst, int64(t.Y), 10)
	dst = append(dst, `,"dir":"`...)
	dst = append(dst, d.String()...)
	return append(dst, '"')
}

func appendTenth(dst []byte, v float64) []byte {
	return strconv.AppendFloat(dst, math.Round(v*10)/10, 'f', -1, 64)
}

// appendString writes s as a JSON string; UTF-8 passes through as it is.
func appendString[S ~string | ~[]byte](dst []byte, s S) []byte {
	dst = append(dst, '"')
	for i := 0; i < len(s); i++ {
		switch c := s[i]; {
		case c == '"' || c == '\\':
			dst = append(dst, '\\', c)
		case c == '\n':
			dst = append(dst, '\\', 'n')
		case c < 0x20:
			dst = append(dst, `\u00`...)
			dst = append(dst, "0123456789abcdef"[c>>4], "0123456789abcdef"[c&15])
		default:
			dst = append(dst, c)
		}
	}
	return append(dst, '"')
}

// Question writes p as `"id":{"type":"choice",...}`.
func (g *Game) Question(dst []byte, p *Point) []byte {
	opts := g.Options(p)
	dst = appendString(dst, p.QuestionID())
	dst = append(dst, `:{"type":"choice","instructions":`...)
	var buf [1024]byte
	dst = appendString(dst, g.instructions(buf[:0], p, opts))
	dst = append(dst, `,"criteria":{`...)
	fruit := NoDir
	if p.Actor == Pacman {
		fruit = g.FruitRoute(p, opts)
	}
	for i := range opts {
		if i > 0 {
			dst = append(dst, ',')
		}
		dst = appendString(dst, opts[i].Dir.String())
		dst = append(dst, ':')
		dst = appendString(dst, g.criterion(buf[:0], p, &opts[i], fruit))
	}
	return append(dst, "}}"...)
}

func names(dst []byte, mask uint8) []byte {
	first := true
	for id := Blinky; id < NActors; id++ {
		if mask&(1<<id) == 0 {
			continue
		}
		if !first {
			dst = append(dst, " and "...)
		}
		first = false
		dst = append(dst, Names[id]...)
	}
	return dst
}

func count(mask uint8) int {
	n := 0
	for ; mask != 0; mask &= mask - 1 {
		n++
	}
	return n
}

func (g *Game) instructions(b []byte, p *Point, opts []Option) []byte {
	where := func(b []byte) []byte {
		if p.Escape {
			b = append(b, "You are heading "...)
			b = append(b, p.Heading.String()...)
			b = append(b, " down a corridor, and "...)
			b = names(b, p.Threats)
			if count(p.Threats) == 1 {
				b = append(b, " is"...)
			} else {
				b = append(b, " are"...)
			}
			b = append(b, " in it ahead of you or can get to its far end first. Carry on "...)
			b = append(b, p.Heading.String()...)
			b = append(b, ", or turn round and go "...)
			b = append(b, p.Heading.Reverse().String()...)
			return append(b, " now?"...)
		}
		b = append(b, "You are coming up to the junction at ("...)
		b = strconv.AppendInt(b, int64(p.Tile.X), 10)
		b = append(b, ',')
		b = strconv.AppendInt(b, int64(p.Tile.Y), 10)
		b = append(b, ") going "...)
		b = append(b, p.Heading.String()...)
		return append(b, ". Choose which way to leave it."...)
	}
	if p.Actor == Pacman {
		b = append(b, "You are Pac-Man. "...)
		if g.FrightLeft > 0 {
			b = append(b, "A power pellet is on for "...)
			b = appendTenth(b, g.FrightLeft)
			b = append(b, " more seconds: the frightened ghosts (lowercase letters) are worth 200, 400, 800 and 1600 in a row. Go for the nearest one you can reach in time, but never into a ghost that is not frightened. "...)
		} else {
			b = append(b, "Eat every pellet to clear the maze and keep clear of the ghosts: touching one that is not frightened costs a life. A power pellet (o) makes the ghosts frightened and edible, so take one when they close in. Avoid a way marked DANGER or TRAP unless every way is; prefer ways where the ghosts are moving off. "...)
		}
		if g.Fruit.On {
			if fr := g.FruitRoute(p, opts); fr != NoDir {
				b = append(b, "There is a "...)
				b = append(b, g.Fruit.Kind...)
				b = append(b, " worth "...)
				b = strconv.AppendInt(b, int64(g.Fruit.Points), 10)
				b = append(b, " points on the board for a short time; the way marked FRUIT is the quickest safe-looking way to it, take it. "...)
			} else if !p.Escape {
				b = append(b, "There is fruit on the board but no safe way gets there in time; leave it. "...)
			}
		}
		return where(b)
	}
	gh := &g.Actors[p.Actor]
	if gh.State == Frightened {
		b = append(b, "You are "...)
		b = append(b, Names[p.Actor]...)
		b = append(b, ", a ghost, frightened for "...)
		b = appendTenth(b, g.FrightLeft)
		b = append(b, " more seconds: Pac-Man can eat you. Run: take the way that keeps you farthest from him. "...)
		return where(b)
	}
	b = append(b, personas[p.Actor]...)
	if g.Mode == Scatter {
		b = append(b, " It is scatter time: go back to your home corner rather than chase."...)
	}
	b = append(b, ' ')
	return where(b)
}

func stepsText(b []byte, n int, what string) []byte {
	b = append(b, what...)
	if n == Unknown {
		return append(b, " not reachable this way"...)
	}
	b = append(b, ' ')
	b = strconv.AppendInt(b, int64(n), 10)
	if n == 1 {
		return append(b, " step away"...)
	}
	return append(b, " steps away"...)
}

func (g *Game) criterion(b []byte, p *Point, o *Option, fruit Dir) []byte {
	b = append(b, "Go "...)
	b = append(b, o.Dir.String()...)
	b = append(b, ": "...)
	if p.Actor == Pacman {
		if fruit == o.Dir && g.Fruit.On {
			b = append(b, "FRUIT, the quickest safe-looking way to the "...)
			b = append(b, g.Fruit.Kind...)
			b = append(b, " ("...)
			b = strconv.AppendInt(b, int64(o.Fruit), 10)
			b = append(b, " steps, "...)
			b = strconv.AppendInt(b, int64(g.Fruit.Points), 10)
			b = append(b, " points); "...)
		}
		b = stepsText(b, o.Pellet, "nearest pellet")
		b = append(b, "; "...)
		b = strconv.AppendInt(b, int64(o.CorridorPellets), 10)
		if o.CorridorPellets == 1 {
			b = append(b, " pellet in the next corridor; "...)
		} else {
			b = append(b, " pellets in the next corridor; "...)
		}
		if o.Danger == Unknown {
			b = append(b, "no dangerous ghost reachable this way"...)
		} else {
			b = stepsText(b, o.Danger, "nearest dangerous ghost")
			if o.Approaching {
				b = append(b, ", coming at you"...)
			} else {
				b = append(b, ", moving off"...)
			}
		}
		if o.Nearby >= 2 {
			b = append(b, "; "...)
			b = strconv.AppendInt(b, int64(o.Nearby), 10)
			b = append(b, " dangerous ghosts within 8 steps"...)
		}
		if o.DangerIn != 0 {
			b = append(b, "; DANGER: "...)
			b = names(b, o.DangerIn)
			b = append(b, " in this corridor"...)
		}
		if o.Trap() {
			b = append(b, "; TRAP: "...)
			b = append(b, Names[o.JunctionGhost]...)
			b = append(b, " reaches the next junction in "...)
			b = strconv.AppendInt(b, int64(o.GhostSteps), 10)
			b = append(b, " steps, you need "...)
			b = strconv.AppendInt(b, int64(o.JunctionSteps), 10)
		} else if o.JunctionGhost >= 0 {
			b = append(b, "; you reach the next junction in "...)
			b = strconv.AppendInt(b, int64(o.JunctionSteps), 10)
			b = append(b, " steps, "...)
			b = strconv.AppendInt(b, int64(o.GhostSteps-o.JunctionSteps), 10)
			b = append(b, " ahead of any ghost"...)
		}
		if g.FrightLeft == 0 && o.Power != Unknown {
			b = append(b, "; "...)
			b = stepsText(b, o.Power, "power pellet")
		}
		if g.FrightLeft > 0 && o.Frightened != Unknown {
			b = append(b, "; "...)
			b = stepsText(b, o.Frightened, "frightened ghost")
		}
		return b
	}
	kind, _, _ := g.Goal(p.Actor)
	if kind == GoalFlee {
		b = stepsText(b, o.ToPacman, "Pac-Man")
		return append(b, " this way (farther is safer)"...)
	}
	b = stepsText(b, o.Goal, kind.String())
	if kind != GoalPacman {
		b = append(b, "; "...)
		b = stepsText(b, o.ToPacman, "Pac-Man")
	}
	if o.DangerIn != 0 {
		b = append(b, "; passes "...)
		b = names(b, o.DangerIn)
	}
	return b
}

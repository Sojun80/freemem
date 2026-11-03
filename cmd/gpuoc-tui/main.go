// gpuoc-tui is a simulation-only NVIDIA tuning interface.
// No key path in this program writes GPU settings. Apply commits only the
// in-memory simulated state so the UI can be developed safely.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

const version = "1.0.0"

type theme string

const (
	matrix theme = "matrix"
	amber  theme = "amber"
)

type preferences struct {
	Theme theme `json:"theme"`
}
type tuning struct {
	Core, Memory, VTT, Fan, MemLock int
	Power                           float64
}
type gpu struct {
	Index                                                 int
	Name, Bus, Temp, MemoryTemp, Draw, Limit, Fan, PState string
	DefaultPower                                          float64
}
type app struct {
	theme           theme
	refresh         time.Duration
	gpus            []gpu
	selected, field int
	settings        bool
	current, target map[int]tuning
	notice          string
	quit            bool
}

func main() {
	refresh := flag.Duration("refresh", time.Second, "telemetry refresh interval")
	flag.DurationVar(refresh, "r", time.Second, "telemetry refresh interval")
	themeFlag := flag.String("theme", "", "initial theme: matrix or amber")
	dryRun := flag.Bool("dry-run", false, "accepted for compatibility; UI is always simulated")
	showVersion := flag.Bool("version", false, "show version")
	flag.BoolVar(showVersion, "V", false, "show version")
	flag.Usage = usage
	flag.Parse()
	if *showVersion {
		fmt.Println(version)
		return
	}
	if *refresh <= 0 {
		die("Refresh interval must be positive.")
	}
	if !isTerminal() {
		die("gpuoc-tui needs an interactive terminal.")
	}
	pref := loadPreferences()
	chosen := pref.Theme
	if *themeFlag != "" {
		chosen = theme(*themeFlag)
	}
	if chosen != matrix && chosen != amber {
		die("Unknown theme: %s (use matrix or amber).", chosen)
	}
	_ = dryRun // all paths are permanently simulation-only
	a := &app{theme: chosen, refresh: *refresh, current: map[int]tuning{}, target: map[int]tuning{}}
	a.refreshGPUs()
	if len(a.gpus) == 0 {
		die("No NVIDIA GPUs were reported by nvidia-smi.")
	}
	a.notice = "Simulation only: F1 Matrix, F2 Amber. No hardware writes are possible."
	run(a)
}

func usage() {
	fmt.Fprint(os.Stdout, `Usage: gpuoc-tui [OPTIONS]

Keyboard-driven NVIDIA GPU tuning prototype. It is permanently simulation-only.

Options:
  -r, --refresh DURATION  Telemetry refresh interval (default: 1s)
      --theme NAME        Initial theme: matrix or amber
      --dry-run           Accepted for compatibility; no writes are possible
  -V, --version           Show version
  -h, --help              Show this help

Keys:
  F1 Matrix green theme (saved)       F2 amber theme (saved)
  Up/Down select GPU or setting       Enter edit / Apply / Reset
  Left/Right adjust staged value      a Apply staged simulation
  r stage reset                       Tab next GPU
  Esc return to GPU list              q quit

Theme is saved in $XDG_CONFIG_HOME/gpuoc-tui/preferences.json (or ~/.config).
`)
}

func run(a *app) {
	state, err := stty("-g")
	if err != nil {
		die("Could not read terminal settings.")
	}
	if _, err := stty("raw", "-echo", "min", "0", "time", "1"); err != nil {
		die("Could not configure terminal: %v", err)
	}
	defer func() { _, _ = stty(state); fmt.Print("\033[?25h\033[0m\033[?1049l\n") }()
	fmt.Print("\033[?1049h\033[?25l")
	keys := make(chan string, 32)
	go readKeys(keys)
	ticker := time.NewTicker(a.refresh)
	defer ticker.Stop()
	for !a.quit {
		a.draw()
		select {
		case key := <-keys:
			a.key(key)
		case <-ticker.C:
			a.refreshGPUs()
		}
	}
}

func (a *app) refreshGPUs() {
	path, err := exec.LookPath("nvidia-smi")
	if err != nil {
		a.notice = "nvidia-smi is required."
		return
	}
	args := []string{"--query-gpu=index,name,pci.bus_id,temperature.gpu,temperature.memory,power.draw,power.limit,power.default_limit,fan.speed,pstate", "--format=csv,noheader,nounits"}
	out, err := exec.Command(path, args...).Output()
	if err != nil {
		a.notice = "Unable to read telemetry from nvidia-smi."
		return
	}
	var next []gpu
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		p := fields(line, 10)
		if p == nil {
			continue
		}
		idx, err := strconv.Atoi(p[0])
		if err != nil {
			continue
		}
		def, _ := strconv.ParseFloat(p[7], 64)
		g := gpu{idx, trimName(p[1]), p[2], p[3], p[4], p[5], p[6], p[8], p[9], def}
		next = append(next, g)
		if _, ok := a.current[idx]; !ok {
			a.current[idx] = tuning{Power: def}
			a.target[idx] = a.current[idx]
		}
	}
	sort.Slice(next, func(i, j int) bool { return next[i].Index < next[j].Index })
	a.gpus = next
	if a.selected >= len(next) {
		a.selected = 0
	}
}

func (a *app) draw() {
	c := colors(a.theme)
	fmt.Print("\033[H\033[2J")
	fmt.Printf("%s GPUOC TUI v%s  %s%s\n", c.title, version, a.themeLabel(), c.reset)
	fmt.Printf("%s F1 Matrix   F2 Amber   Theme is saved automatically   Simulation only %s\n\n", c.dim, c.reset)
	if !a.settings {
		a.drawPicker(c)
	} else {
		a.drawSettings(c)
	}
	fmt.Printf("\n%s%s%s\n", c.notice, a.notice, c.reset)
}

func (a *app) drawPicker(c palette) {
	fmt.Printf("%sSelect a GPU — Enter locks it for editing%s\n", c.title, c.reset)
	for i, g := range a.gpus {
		mark := " "
		color := c.dim
		if i == a.selected {
			mark = ">"
			color = c.selected
		}
		fmt.Printf("%s%s GPU %d  %-22s  %3sC  %s/%s W  fan %s%%  %s%s\n", color, mark, g.Index, g.Name, g.Temp, g.Draw, g.Limit, g.Fan, g.PState, c.reset)
	}
	fmt.Printf("\n%sUp/Down select • Enter settings • q quit%s\n", c.dim, c.reset)
}

func (a *app) drawSettings(c palette) {
	g := a.gpus[a.selected]
	cur, target := a.current[g.Index], a.target[g.Index]
	fmt.Printf("%sGPU %d — %s  (locked selection)%s\n", c.title, g.Index, g.Name, c.reset)
	fmt.Printf("%sLive: %sC GPU • %sC memory • %s/%s W • fan %s%% • %s%s\n\n", c.dim, g.Temp, g.MemoryTemp, g.Draw, g.Limit, g.Fan, g.PState, c.reset)
	rows := []struct{ name, value string }{
		{"Core offset", signed(target.Core) + " MHz"}, {"Memory offset", signed(target.Memory) + " MHz"},
		{"Power target", fmt.Sprintf("%.0f W", target.Power)}, {"Fan target", fanText(target.Fan)},
		{"VTT offset", signed(target.VTT) + " mV"}, {"Memory lock", lockText(target.MemLock)},
		{"[Apply]", "commit staged values to simulation"}, {"[Reset]", "stage stock values"},
	}
	for i, row := range rows {
		color := c.dim
		mark := " "
		if i == a.field {
			color = c.selected
			mark = ">"
		}
		delta := ""
		if i < 6 && !equalField(i, cur, target) {
			delta = " *"
		}
		fmt.Printf("%s%s %-16s %-34s%s%s\n", color, mark, row.name, row.value, delta, c.reset)
	}
	fmt.Printf("\n%sLeft/Right adjust • Enter action • a apply • r reset • Esc back%s\n", c.dim, c.reset)
}

func (a *app) themeLabel() string {
	if a.theme == amber {
		return "AMBER"
	}
	return "MATRIX"
}
func (a *app) key(k string) {
	if k == "F1" {
		a.setTheme(matrix)
		return
	}
	if k == "F2" {
		a.setTheme(amber)
		return
	}
	if k == "q" || k == "Q" {
		a.quit = true
		return
	}
	if len(a.gpus) == 0 {
		return
	}
	if !a.settings {
		switch k {
		case "UP", "k":
			a.selected = (a.selected + len(a.gpus) - 1) % len(a.gpus)
		case "DOWN", "j":
			a.selected = (a.selected + 1) % len(a.gpus)
		case "ENTER":
			a.settings = true
			a.field = 0
			a.notice = "GPU locked; tune targets, then Apply simulates the commit."
		}
		return
	}
	switch k {
	case "ESC":
		a.settings = false
		a.notice = "Returned to GPU list."
	case "TAB":
		a.selected = (a.selected + 1) % len(a.gpus)
		a.notice = "Switched locked GPU."
	case "UP", "k":
		a.field = (a.field + 7) % 8
	case "DOWN", "j":
		a.field = (a.field + 1) % 8
	case "LEFT", "h", "-":
		a.adjust(-1)
	case "RIGHT", "l", "+", "=":
		a.adjust(1)
	case "ENTER":
		if a.field == 6 {
			a.apply()
		} else if a.field == 7 {
			a.reset()
		}
	case "a", "A":
		a.apply()
	case "r", "R":
		a.reset()
	}
}

func (a *app) setTheme(next theme) {
	a.theme = next
	if err := savePreferences(preferences{Theme: next}); err != nil {
		a.notice = "Theme changed, but could not save preference: " + err.Error()
	} else {
		a.notice = "Theme set to " + string(next) + " and saved."
	}
}
func (a *app) adjust(direction int) {
	g := a.gpus[a.selected]
	t := a.target[g.Index]
	switch a.field {
	case 0:
		t.Core += direction * 25
	case 1:
		if t.MemLock == 0 {
			t.Memory += direction * 50
		} else {
			a.notice = "Memory offset is disabled while a fixed memory clock is selected."
			return
		}
	case 2:
		t.Power += float64(direction * 5)
		if t.Power < 1 {
			t.Power = 1
		}
	case 3:
		t.Fan += direction * 5
		if t.Fan < 0 {
			t.Fan = 0
		}
		if t.Fan > 100 {
			t.Fan = 100
		}
	case 4:
		t.VTT += direction * 5
	case 5:
		t.MemLock = nextMemLock(g.Name, t.MemLock, direction)
	default:
		a.notice = "Use Enter for Apply or Reset."
		return
	}
	a.target[g.Index] = t
	a.notice = "Staged target changed. Hardware is unchanged."
}
func (a *app) apply() {
	g := a.gpus[a.selected]
	if a.current[g.Index] == a.target[g.Index] {
		a.notice = "There are no pending simulated changes."
		return
	}
	a.current[g.Index] = a.target[g.Index]
	a.notice = "Simulated apply complete. Hardware unchanged."
}
func (a *app) reset() {
	g := a.gpus[a.selected]
	a.target[g.Index] = tuning{Power: g.DefaultPower}
	a.notice = "Stock targets staged. Select Apply to commit the simulation."
}

func nextMemLock(name string, current, direction int) int {
	choices := []int{0, 810}
	upper := strings.ToUpper(name)
	if strings.Contains(upper, "RTX 30") || strings.Contains(upper, "RTX 40") {
		choices = []int{0, 810, 5000, 5001, 6000}
	}
	pos := 0
	for i, v := range choices {
		if v == current {
			pos = i
		}
	}
	return choices[(pos+direction+len(choices))%len(choices)]
}
func equalField(field int, a, b tuning) bool {
	switch field {
	case 0:
		return a.Core == b.Core
	case 1:
		return a.Memory == b.Memory
	case 2:
		return a.Power == b.Power
	case 3:
		return a.Fan == b.Fan
	case 4:
		return a.VTT == b.VTT
	case 5:
		return a.MemLock == b.MemLock
	}
	return true
}
func fields(line string, count int) []string {
	p := strings.Split(line, ",")
	if len(p) != count {
		return nil
	}
	for i := range p {
		p[i] = strings.TrimSpace(p[i])
	}
	return p
}
func trimName(s string) string {
	return strings.TrimPrefix(strings.TrimPrefix(s, "NVIDIA "), "GeForce ")
}
func signed(n int) string {
	if n > 0 {
		return "+" + strconv.Itoa(n)
	}
	return strconv.Itoa(n)
}
func fanText(n int) string {
	if n == 0 {
		return "AUTO"
	}
	return strconv.Itoa(n) + "%"
}
func lockText(n int) string {
	if n == 0 {
		return "DEFAULT (unlocked)"
	}
	return strconv.Itoa(n) + " MHz (offset disabled)"
}
func stty(args ...string) (string, error) {
	cmd := exec.Command("stty", args...)
	cmd.Stdin = os.Stdin
	out, err := cmd.Output()
	return strings.TrimSpace(string(out)), err
}
func isTerminal() bool {
	return os.Getenv("TERM") != "" && fileTerminal(os.Stdin) && fileTerminal(os.Stdout)
}
func fileTerminal(f *os.File) bool {
	info, err := f.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}
func die(format string, args ...any) { fmt.Fprintf(os.Stderr, format+"\n", args...); os.Exit(2) }

func preferencePath() string {
	base := os.Getenv("XDG_CONFIG_HOME")
	if base == "" {
		home, _ := os.UserHomeDir()
		base = filepath.Join(home, ".config")
	}
	return filepath.Join(base, "gpuoc-tui", "preferences.json")
}
func loadPreferences() preferences {
	data, err := os.ReadFile(preferencePath())
	if err != nil {
		return preferences{Theme: matrix}
	}
	var p preferences
	if json.Unmarshal(data, &p) != nil || (p.Theme != matrix && p.Theme != amber) {
		return preferences{Theme: matrix}
	}
	return p
}
func savePreferences(p preferences) error {
	path := preferencePath()
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	temp := path + ".tmp"
	if err = os.WriteFile(temp, append(data, '\n'), 0600); err != nil {
		return err
	}
	return os.Rename(temp, path)
}

type palette struct{ title, selected, dim, notice, reset string }

func colors(t theme) palette {
	if t == amber {
		return palette{"\033[38;5;208m", "\033[38;5;202m", "\033[38;5;130m", "\033[38;5;172m", "\033[0m"}
	}
	return palette{"\033[36m", "\033[92m", "\033[2m", "\033[33m", "\033[0m"}
}

func readKeys(out chan<- string) {
	b := make([]byte, 1)
	for {
		n, err := os.Stdin.Read(b)
		if err != nil {
			return
		}
		if n == 0 {
			continue
		}
		if b[0] != 27 {
			out <- normalKey(b[0])
			continue
		}
		time.Sleep(8 * time.Millisecond)
		tail := make([]byte, 8)
		n, _ = os.Stdin.Read(tail)
		seq := string(tail[:n])
		switch seq {
		case "[A":
			out <- "UP"
		case "[B":
			out <- "DOWN"
		case "[C":
			out <- "RIGHT"
		case "[D":
			out <- "LEFT"
		case "OP":
			out <- "F1"
		case "OQ":
			out <- "F2"
		case "[11~":
			out <- "F1"
		case "[12~":
			out <- "F2"
		default:
			out <- "ESC"
		}
	}
}
func normalKey(b byte) string {
	switch b {
	case 13, 10:
		return "ENTER"
	case 9:
		return "TAB"
	case 3:
		return "q"
	}
	return string(b)
}

// gpuoc is the safe NVIDIA control backend for the gpuoc TUI.
//
// It intentionally has no hardware-write implementation yet.  `apply` and
// `reset` validate their input and require --dry-run, where they print the
// exact driver commands that would be used by a future write adapter.
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
)

const version = "0.2.0"

type gpu struct {
	index                            int
	name, bus                        string
	defaultPower, minPower, maxPower float64
}

type controls struct {
	core, memory, fan, vtt *int
	power                  *float64
}

type options struct {
	gpu     *int
	all     bool
	dryRun  bool
	yes     bool
	unsafe  bool
	control controls
}

func usage() {
	fmt.Fprintf(os.Stdout, `Usage:
  gpuoc status [--gpu INDEX]
  gpuoc apply --gpu INDEX [CONTROLS] [--yes|--dry-run]
  gpuoc apply --all [CONTROLS] [--yes|--dry-run]
  gpuoc reset --gpu INDEX [--yes|--dry-run]
  gpuoc reset --all [--yes|--dry-run]

NVIDIA GPU telemetry and guarded control command preview.

Hardware writes are disabled. apply/reset only work with --dry-run, which
prints the proposed nvidia-settings/nvidia-smi commands without executing them.

Controls for apply:
  --core OFFSET_MHZ       Graphics clock offset (default cap: +/-500 MHz)
  --memory OFFSET_MHZ     Memory transfer-rate offset (default cap: +/-3000 MHz)
  --power-limit W         Power target within the driver's reported range
  --fan PERCENT           Manual fan target from 20 through 100 percent
  --vtt OFFSET_MV         Voltage offset (default cap: +/-50 mV)

Options:
  -g, --gpu INDEX         Select one nvidia-smi GPU
      --all               Explicitly select every GPU
      --yes               Accepted for API compatibility; cannot enable writes
      --dry-run           Print proposed writes without executing them
      --unsafe            Permit offsets beyond conservative caps
  -h, --help              Show this help
      --version           Show the tool version
`)
}

func main() {
	if len(os.Args) == 1 {
		runStatus(nil)
		return
	}
	if os.Args[1] == "--help" || os.Args[1] == "-h" {
		usage()
		return
	}
	if os.Args[1] == "--version" || os.Args[1] == "-V" {
		fmt.Println(version)
		return
	}

	command := os.Args[1]
	if command != "status" && command != "apply" && command != "reset" {
		fail("Unknown command: %s (try --help).", command)
	}
	opts := parse(command, os.Args[2:])
	if command == "status" {
		if opts.all || hasControls(opts.control) {
			fail("status accepts only --gpu INDEX.")
		}
		runStatus(opts.gpu)
		return
	}
	if opts.gpu == nil && !opts.all {
		fail("%s requires --gpu INDEX or --all.", command)
	}
	if !opts.dryRun {
		fail("GPU hardware writes are disabled; use --dry-run to preview commands.")
	}
	gpus := inventory()
	targets := selectGPUs(gpus, opts.gpu, opts.all)
	if command == "apply" {
		apply(targets, opts)
		return
	}
	reset(targets)
}

func parse(command string, args []string) options {
	fs := flag.NewFlagSet("gpuoc "+command, flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var gpuText string
	var o options
	fs.StringVar(&gpuText, "gpu", "", "GPU index")
	fs.StringVar(&gpuText, "g", "", "GPU index")
	fs.BoolVar(&o.all, "all", false, "all GPUs")
	fs.BoolVar(&o.dryRun, "dry-run", false, "preview only")
	fs.BoolVar(&o.yes, "yes", false, "accepted for compatibility")
	fs.BoolVar(&o.unsafe, "unsafe", false, "allow larger offsets")
	var core, memory, fan, vtt string
	var power string
	fs.StringVar(&core, "core", "", "core offset")
	fs.StringVar(&memory, "memory", "", "memory offset")
	fs.StringVar(&power, "power-limit", "", "power limit")
	fs.StringVar(&fan, "fan", "", "fan speed")
	fs.StringVar(&vtt, "vtt", "", "voltage offset")
	fs.Usage = usage
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	if fs.NArg() != 0 {
		fail("Unknown option: %s (try --help).", fs.Arg(0))
	}
	if o.all && gpuText != "" {
		fail("--gpu and --all cannot be used together.")
	}
	if gpuText != "" {
		n, err := nonNegativeInt(gpuText)
		if err != nil {
			fail("GPU index must be a non-negative integer.")
		}
		o.gpu = &n
	}
	o.control.core = optionalInt(core, "--core must be an integer number of MHz.")
	o.control.memory = optionalInt(memory, "--memory must be an integer number of MHz.")
	o.control.fan = optionalInt(fan, "--fan must be an integer percentage.")
	o.control.vtt = optionalInt(vtt, "--vtt must be an integer number of millivolts.")
	if power != "" {
		n, err := strconv.ParseFloat(power, 64)
		if err != nil {
			fail("--power-limit must be a number in watts.")
		}
		o.control.power = &n
	}
	return o
}

func optionalInt(text, message string) *int {
	if text == "" {
		return nil
	}
	n, err := strconv.Atoi(text)
	if err != nil {
		fail(message)
	}
	return &n
}

func nonNegativeInt(text string) (int, error) {
	n, err := strconv.Atoi(text)
	if err != nil || n < 0 {
		return 0, errors.New("invalid")
	}
	return n, nil
}
func hasControls(c controls) bool {
	return c.core != nil || c.memory != nil || c.power != nil || c.fan != nil || c.vtt != nil
}

func inventory() []gpu {
	rows := smi("--query-gpu=index,name,pci.bus_id,power.default_limit,power.min_limit,power.max_limit", "--format=csv,noheader,nounits")
	var out []gpu
	for _, row := range rows {
		parts := csvFields(row, 6)
		if parts == nil {
			continue
		}
		index, err := nonNegativeInt(parts[0])
		if err != nil {
			continue
		}
		def, e1 := strconv.ParseFloat(parts[3], 64)
		min, e2 := strconv.ParseFloat(parts[4], 64)
		max, e3 := strconv.ParseFloat(parts[5], 64)
		if e1 != nil || e2 != nil || e3 != nil {
			fail("nvidia-smi returned unusable power limits for GPU %d.", index)
		}
		out = append(out, gpu{index: index, name: parts[1], bus: parts[2], defaultPower: def, minPower: min, maxPower: max})
	}
	if len(out) == 0 {
		fail("nvidia-smi returned no usable GPU records.")
	}
	sort.Slice(out, func(i, j int) bool { return out[i].index < out[j].index })
	return out
}

func smi(args ...string) []string {
	path, err := exec.LookPath("nvidia-smi")
	if err != nil {
		fail("nvidia-smi is required.")
	}
	out, err := exec.Command(path, args...).Output()
	if err != nil {
		fail("Unable to read GPU data from nvidia-smi.")
	}
	return strings.Split(strings.TrimSpace(string(out)), "\n")
}

func csvFields(row string, count int) []string {
	parts := strings.Split(row, ",")
	if len(parts) != count {
		return nil
	}
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	return parts
}

func selectGPUs(gpus []gpu, desired *int, all bool) []gpu {
	if all {
		return gpus
	}
	for _, g := range gpus {
		if g.index == *desired {
			return []gpu{g}
		}
	}
	fail("GPU index %d was not reported by nvidia-smi.", *desired)
	return nil
}

func apply(gpus []gpu, o options) {
	if !hasControls(o.control) {
		fail("apply needs at least one of --core, --memory, --power-limit, --fan, or --vtt.")
	}
	validate(o.control, o.unsafe)
	for _, g := range gpus {
		if o.control.power != nil && (*o.control.power < g.minPower || *o.control.power > g.maxPower) {
			fail("GPU %d power limit must be between %.0f W and %.0f W.", g.index, g.minPower, g.maxPower)
		}
		settingsGPU := -1
		needsSettings := o.control.core != nil || o.control.memory != nil || o.control.vtt != nil || o.control.fan != nil
		if needsSettings {
			settingsGPU = settingsIndex(g)
		}
		if o.control.core != nil {
			previewSettings(settingsGPU, "GPUGraphicsClockOffsetAllPerformanceLevels", *o.control.core)
		}
		if o.control.memory != nil {
			previewSettings(settingsGPU, "GPUMemoryTransferRateOffsetAllPerformanceLevels", *o.control.memory)
		}
		if o.control.vtt != nil {
			previewSettings(settingsGPU, "GPUOverVoltageOffset", *o.control.vtt*1000)
		}
		if o.control.power != nil {
			fmt.Printf("DRY RUN nvidia-smi -i %d -pl %s\n", g.index, fmtNumber(*o.control.power))
		}
		if o.control.fan != nil {
			previewSettings(settingsGPU, "GPUFanControlState", 1)
			previewSettings(settingsGPU, "GPUTargetFanSpeed", *o.control.fan)
		}
	}
}

func reset(gpus []gpu) {
	for _, g := range gpus {
		settingsGPU := settingsIndex(g)
		previewSettings(settingsGPU, "GPUGraphicsClockOffsetAllPerformanceLevels", 0)
		previewSettings(settingsGPU, "GPUMemoryTransferRateOffsetAllPerformanceLevels", 0)
		previewSettings(settingsGPU, "GPUOverVoltageOffset", 0)
		previewSettings(settingsGPU, "GPUFanControlState", 0)
		fmt.Printf("DRY RUN nvidia-smi -i %d -pl %s\n", g.index, fmtNumber(g.defaultPower))
	}
}

func previewSettings(settingsGPU int, attribute string, value int) {
	fmt.Printf("DRY RUN nvidia-settings -a [gpu:%d]/%s=%d\n", settingsGPU, attribute, value)
}

// nvidia-smi and nvidia-settings may enumerate GPUs differently.  Map via
// PCI bus rather than assuming their numeric indexes agree.
func settingsIndex(g gpu) int {
	path, err := exec.LookPath("nvidia-settings")
	if err != nil {
		fail("GPU %d requires nvidia-settings, which is not installed.", g.index)
	}
	target, ok := pciBusNumber(g.bus)
	if !ok {
		fail("GPU %d has an unusable PCI bus id %q.", g.index, g.bus)
	}
	for index := 0; index < 32; index++ {
		out, err := exec.Command(path, "-q", fmt.Sprintf("[gpu:%d]/PCIBus", index), "-t").Output()
		if err != nil {
			continue
		}
		value := strings.TrimSpace(string(out))
		if strings.HasPrefix(value, "0x") || strings.HasPrefix(value, "0X") {
			value = value[2:]
		}
		bus, err := strconv.ParseInt(value, 0, 64)
		if err == nil && int(bus) == target {
			return index
		}
	}
	fail("GPU %d could not be mapped to an nvidia-settings GPU; check DISPLAY/XAUTHORITY.", g.index)
	return -1
}

func pciBusNumber(bus string) (int, bool) {
	parts := strings.Split(bus, ":")
	if len(parts) < 2 {
		return 0, false
	}
	n, err := strconv.ParseInt(parts[len(parts)-2], 16, 64)
	return int(n), err == nil
}

func validate(c controls, unsafe bool) {
	if c.core != nil && !unsafe && abs(*c.core) > 500 {
		fail("--core is capped at +/-500 MHz; add --unsafe to override the conservative cap.")
	}
	if c.memory != nil && !unsafe && abs(*c.memory) > 3000 {
		fail("--memory is capped at +/-3000 MHz; add --unsafe to override the conservative cap.")
	}
	if c.vtt != nil && !unsafe && abs(*c.vtt) > 50 {
		fail("--vtt is capped at +/-50 mV; add --unsafe to override the conservative cap.")
	}
	if c.fan != nil && (*c.fan < 20 || *c.fan > 100) {
		fail("--fan must be between 20 and 100 percent.")
	}
}

func runStatus(filter *int) {
	rows := smi("--query-gpu=index,name,pci.bus_id,temperature.gpu,temperature.memory,power.draw,power.limit,memory.used,memory.total,clocks.current.graphics,clocks.current.memory,fan.speed,pstate", "--format=csv,noheader,nounits")
	fmt.Printf("GPUOC v%s  NVIDIA control status\n\n", version)
	fmt.Printf("%-4s %-25s %7s %7s %11s %15s %5s %6s %5s %4s\n", "GPU", "MODEL", "TEMP", "MEM T", "POWER", "VRAM", "GCLK", "MCLK", "FAN", "P")
	for _, row := range rows {
		p := csvFields(row, 13)
		if p == nil {
			continue
		}
		index, err := nonNegativeInt(p[0])
		if err != nil || (filter != nil && index != *filter) {
			continue
		}
		name := strings.TrimPrefix(strings.TrimPrefix(p[1], "NVIDIA "), "GeForce ")
		fmt.Printf("%-4d %-25.25s %6sC %6sC %6s/%-4sW %6s/%-6s %5s %6s %5s %4s\n", index, name, p[3], p[4], p[5], p[6], gib(p[7]), gib(p[8]), p[9], p[10], percent(p[11]), p[12])
	}
	fmt.Println("\nHardware writes are disabled; apply/reset only print commands with --dry-run.")
}

func gib(value string) string {
	n, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return "N/A"
	}
	return fmt.Sprintf("%.1fG", n/1024)
}
func percent(value string) string {
	if value == "[N/A]" || value == "N/A" {
		return "N/A"
	}
	return value + "%"
}
func fmtNumber(v float64) string { return strconv.FormatFloat(v, 'f', -1, 64) }
func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
func fail(format string, args ...any) { fmt.Fprintf(os.Stderr, format+"\n", args...); os.Exit(2) }

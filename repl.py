"""
BIB Interactive REPL
=====================
Run as: python repl.py

Simulates a mock organism: you type text/commands,
BIB processes them as sensory bytes and responds with action names.
Displays live chemical bar charts and brain telemetry via Rich.

Commands:
  [REWARD:0.8]   — inject graded reward signal
  [PAIN:0.5]     — inject graded pain signal
  [SLEEP]        — force sleep cycle
  [STATS]        — dump full telemetry
  [SAVE path]    — save brain snapshot
  [LOAD path]    — load brain snapshot
  exit / quit    — exit REPL
"""
from __future__ import annotations

import sys
import re
import time
import numpy as np

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.layout import Layout
    from rich.live import Live
    from rich import box
except ImportError:
    print("Install rich: pip install rich")
    sys.exit(1)

# Add parent to path if running directly
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bib.brain import BIB
from bib.config import ACTION_NAMES, MODALITY_NAMES, MODALITY_INPUT_DIMS

console = Console()

# ── REPL helpers ─────────────────────────────────────────────────────────────

def parse_commands(text: str) -> tuple[str, float, float, bool, float | None, str | None, str | None]:
    """Parse special tags from user input. Returns (clean_text, reward, pain, sleep, save_path, load_path)."""
    reward = 0.0
    pain   = 0.0
    do_sleep = False
    save_path = None
    load_path = None

    reward_match = re.search(r'\[REWARD(?::([0-9.]+))?\]', text, re.IGNORECASE)
    if reward_match:
        reward = float(reward_match.group(1) or "1.0")
        text = re.sub(r'\[REWARD(?::[0-9.]+)?\]', '', text, flags=re.IGNORECASE)

    pain_match = re.search(r'\[PAIN(?::([0-9.]+))?\]', text, re.IGNORECASE)
    if pain_match:
        pain = float(pain_match.group(1) or "1.0")
        text = re.sub(r'\[PAIN(?::[0-9.]+)?\]', '', text, flags=re.IGNORECASE)

    if '[SLEEP]' in text.upper():
        do_sleep = True
        text = re.sub(r'\[SLEEP\]', '', text, flags=re.IGNORECASE)

    save_match = re.search(r'\[SAVE\s+([^\]]+)\]', text, re.IGNORECASE)
    if save_match:
        save_path = save_match.group(1).strip()
        text = re.sub(r'\[SAVE\s+[^\]]+\]', '', text, flags=re.IGNORECASE)

    load_match = re.search(r'\[LOAD\s+([^\]]+)\]', text, re.IGNORECASE)
    if load_match:
        load_path = load_match.group(1).strip()
        text = re.sub(r'\[LOAD\s+[^\]]+\]', '', text, flags=re.IGNORECASE)

    return text.strip(), reward, pain, do_sleep, save_path, load_path


def text_to_sensors(text: str) -> dict[str, np.ndarray]:
    """
    Convert text input to mock sensory vectors for all 6 modalities.
    Vision: ASCII character values → float vector
    Touch: zero (no contact)
    Proprioception: random stub (organism is stationary in REPL)
    Chemo: character frequency features
    Intero: driven by accumulated state
    Auditory: zero
    """
    sensors: dict[str, np.ndarray] = {}

    # Vision: text as raw ASCII byte values
    vision_dim = MODALITY_INPUT_DIMS["vision"]
    vision = np.zeros(vision_dim, dtype=np.float32)
    for i, ch in enumerate(text[:vision_dim]):
        vision[i] = ord(ch) / 255.0
    sensors["vision"] = vision

    # Touch: all zero (no physical contact in text mode)
    sensors["touch"] = np.zeros(MODALITY_INPUT_DIMS["touch"], dtype=np.float32)

    # Proprioception: static stub
    sensors["proprioception"] = np.zeros(MODALITY_INPUT_DIMS["proprioception"], dtype=np.float32)
    sensors["proprioception"][0] = 0.5  # upright posture

    # Chemoreception: character frequency as chemical gradient proxy
    chemo = np.zeros(MODALITY_INPUT_DIMS["chemoreception"], dtype=np.float32)
    for ch in text:
        idx = ord(ch) % MODALITY_INPUT_DIMS["chemoreception"]
        chemo[idx] += 1.0
    if chemo.max() > 0:
        chemo /= chemo.max()
    sensors["chemoreception"] = chemo

    # Interoception: zero (hypothalamus state reflected through homeostatic_state)
    sensors["interoception"] = np.zeros(MODALITY_INPUT_DIMS["interoception"], dtype=np.float32)

    # Auditory: zero
    sensors["auditory"] = np.zeros(MODALITY_INPUT_DIMS["auditory"], dtype=np.float32)

    return sensors


def bar(value: float, width: int = 20, color: str = "green") -> Text:
    """Render a colorful progress bar."""
    filled = int(value * width)
    filled = max(0, min(filled, width))
    bar_str = "█" * filled + "░" * (width - filled)
    return Text(f"[{bar_str}] {value:.2f}", style=color)


def render_chem_panel(chem: dict) -> Panel:
    """Render the chemistry display panel."""
    t = Table.grid(padding=(0, 1))
    t.add_column(style="bold cyan", width=10)
    t.add_column()
    t.add_column(style="dim", width=6)

    chemical_colors = {
        "DA":       "bright_yellow",
        "NE":       "bright_cyan",
        "5-HT":     "bright_magenta",
        "ACh":      "bright_green",
        "Cortisol": "bright_red",
    }
    for name, val in chem.items():
        normalized = (val + 1.0) / 2.0 if name == "DA" else val  # DA is -1 to +1
        color = chemical_colors.get(name, "white")
        t.add_row(name, bar(normalized, width=24, color=color), f"{val:+.3f}")

    return Panel(t, title="[bold]🧬 Neurochemistry[/bold]", border_style="cyan")


def render_homeostasis_panel(h: dict) -> Panel:
    t = Table.grid(padding=(0, 1))
    t.add_column(style="bold", width=10)
    t.add_column()
    drive_colors = {
        "hunger": "orange3", "thirst": "sky_blue2",
        "fatigue": "purple", "pain": "red", "cortisol": "dark_red"
    }
    for key in ["hunger", "thirst", "fatigue", "pain"]:
        val = h.get(key, 0.0)
        color = drive_colors.get(key, "white")
        t.add_row(key.capitalize(), bar(val, width=16, color=color))
    sleep_style = "bold red" if h.get("needs_sleep") else "dim"
    t.add_row("Sleep", Text("⚡ NEEDED" if h.get("needs_sleep") else "OK", style=sleep_style))
    return Panel(t, title="[bold]🫀 Homeostasis[/bold]", border_style="magenta")


def render_memory_panel(tele: dict) -> Panel:
    h = tele.get("hippocampus", {})
    p = tele.get("prefrontal", {})
    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", width=20)
    t.add_column()
    t.add_row("CA3 binds",       str(h.get("binds", 0)))
    t.add_row("CA3 retrievals",  str(h.get("retrievals", 0)))
    t.add_row("Goal cols active", str(p.get("active_goal_cols", 0)))
    t.add_row("WM occupancy",    f"{p.get('wm_occupancy', 0):.1%}")
    t.add_row("PER buffer",      str(tele.get("episodic_buffer_size", 0)))
    return Panel(t, title="[bold]🧠 Memory[/bold]", border_style="green")


# ── Main REPL loop ─────────────────────────────────────────────────────────

def main() -> None:
    console.rule("[bold bright_cyan]BIB — Biologically Inspired Brain[/bold bright_cyan]")
    console.print(
        "[dim]Type anything to send sensory input. Commands: "
        "[REWARD:0.8] [PAIN:0.5] [SLEEP] [STATS] [SAVE path] [LOAD path] exit[/dim]\n"
    )
    console.print("[yellow]⚡ Initializing BIB organs...[/yellow]")
    brain = BIB()
    console.print("[green]✓ BIB ready.[/green]\n")

    tick_count = 0
    last_action_name = "IDLE"

    while True:
        try:
            raw = console.input("[bold bright_white]you › [/bold bright_white]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Shutting down BIB.[/dim]")
            break

        if raw.strip().lower() in ("exit", "quit"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if "[STATS]" in raw.upper():
            tele = brain.get_telemetry()
            console.print_json(data=tele)
            continue

        text, reward, pain, do_sleep, save_path, load_path = parse_commands(raw)

        if save_path:
            brain.save(save_path)
            console.print(f"[green]✓ Brain saved to {save_path}[/green]")
            continue

        if load_path:
            brain.load(load_path)
            console.print(f"[green]✓ Brain loaded from {load_path}[/green]")
            continue

        if do_sleep:
            console.print("[yellow]💤 Running sleep cycle...[/yellow]")
            report = brain.sleep()
            console.print(f"[green]✓ Sleep complete: {report}[/green]")
            continue

        # Mock organism homeostatic state
        homeostatic_state = {
            "hunger": brain.hypothalamus.hunger,
            "thirst": brain.hypothalamus.thirst,
            "pain":   pain,
        }

        # Convert text → sensory vectors
        sensors = text_to_sensors(text or " ")

        # Brain tick
        action = brain.tick(
            sensors=sensors,
            reward=reward,
            homeostatic_state=homeostatic_state,
        )
        tick_count += 1
        last_action_name = brain.get_action_name(action)

        # Get telemetry
        chem = brain.get_chemistry()
        home = brain.hypothalamus.report()
        tele = brain.get_telemetry()

        # Render panels side by side
        console.print()
        console.print(render_chem_panel(chem))
        # Side-by-side homeostasis + memory
        layout = Table.grid(expand=True)
        layout.add_column(ratio=1)
        layout.add_column(ratio=1)
        layout.add_row(render_homeostasis_panel(home), render_memory_panel(tele))
        console.print(layout)

        # Action output
        action_color = {
            "MOVEMENT": "bright_green",
            "NUTRITION": "bright_yellow",
            "REPRODUCTION": "bright_magenta",
            "SENSITIVITY": "bright_cyan",
            "GROWTH": "green",
            "EXCRETION": "yellow",
            "IDLE": "dim",
            "COMMUNICATE": "bright_blue",
        }.get(last_action_name, "white")

        confidence = brain.basal_ganglia.get_confidence(tele.get("last_action", ""))
        console.print(
            f"\n[{action_color}]🧠 ACTION → {last_action_name}[/{action_color}]"
            f"  [dim]tick={tick_count} | TD-error={tele['last_td_error']:+.4f}[/dim]"
        )

        if brain.needs_sleep():
            console.print("[bold red]⚡ BRAIN NEEDS SLEEP — type [SLEEP] to run sleep cycle[/bold red]")

        console.print()


if __name__ == "__main__":
    main()

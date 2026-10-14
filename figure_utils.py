import os
from datetime import datetime


def save_figure(fig, script, name, folder="Figures"):
    """Save fig to Figures/<script>/<name>_<date>_<time>.png so every graph made is kept.
    The timestamp (to the second) gives each run its own file, so nothing is overwritten.
    Call before plt.show(), which clears the figure once the window is closed."""
    out_dir = os.path.join(folder, script)
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = os.path.join(out_dir, f"{name}_{timestamp}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")
    return path

"""architecture_diagram.py - draws Figure 3.1 (proposed MES architecture)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures", "fig3_1_architecture.png")
INK, INK2 = "#0b0b0b", "#52514e"
LAYER = {  # (fill, edge)
    "src": ("#f4f3f0", "#8d8b85"), "edge": ("#e8f1fb", "#2a78d6"), "ing": ("#e6f6f0", "#1baf7a"),
    "sto": ("#fdf1e8", "#eb6834"), "pro": ("#efedfb", "#6250d6"), "con": ("#fbeef3", "#d55181"),
}

fig, ax = plt.subplots(figsize=(13.5, 8.2))
ax.set_xlim(0, 135); ax.set_ylim(0, 82); ax.axis("off")


def box(x, y, w, h, text, kind, fs=8.2, bold=False, alpha=1.0):
    f, e = LAYER[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=f, ec=e, lw=1.3, alpha=alpha))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def column(x, w, title, kind):
    f, e = LAYER[kind]
    ax.add_patch(FancyBboxPatch((x, 17), w, 58, boxstyle="round,pad=0.3,rounding_size=1.8",
                                fc="white", ec=e, lw=1.8))
    ax.text(x + w / 2, 72.6, title, ha="center", va="center", fontsize=9.2, fontweight="bold", color=e)


def arrow(x1, y1, x2, y2, col=INK2, ls="-", lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 color=col, lw=lw, linestyle=ls))


ax.text(67.5, 79.5, "Kappa-style unified processing: one event log and one Spark codebase serve real-time and batch; "
        "batch sources enter as bounded streams. One deployment per jurisdiction (UK region / EU region).",
        fontsize=8.4, color=INK2, ha="center")

# ---- columns
cols = [(1, 18, "1  DATA SOURCES", "src"), (21.5, 19, "2  EDGE", "edge"), (43.5, 20, "3  INGESTION", "ing"),
        (66.5, 22, "4  LAKEHOUSE", "sto"), (91.5, 21, "5  PROCESSING & ML", "pro"),
        (115.5, 18.5, "6  CONSUMPTION", "con")]
for c in cols:
    column(*c)

# sources
for i, t in enumerate(["Smart meters\n(DLMS/COSEM, 3 vendors)", "EV charge points\n(OCPP 2.0.1)",
                       "Solar inverters\n(SunSpec / Modbus)", "Substation IoT / SCADA\n(IEC 61850)",
                       "Weather feeds\n(third-party API)", "Billing platform\n(daily batch)",
                       "Maintenance system\n(daily batch + notes)"]):
    box(2.2, 63 - i * 6.6, 15.6, 5.3, t, "src", fs=6.9)
# edge
box(23, 56, 16, 12, "Substation edge gateway\n• local fault detection\n• store-and-forward\n  buffer\n• IEC 61850 → MQTT", "edge", fs=6.9)
box(23, 41, 16, 12, "Meter head-end /\nvendor adapters\n• protocol & unit\n  translation\n(vendor data contract)", "edge", fs=6.9)
box(23, 26, 16, 12, "Secure file / API\ngateway (batch)\n• SFTP / REST pull\n• checksum & schema\n  check", "edge", fs=6.9)
# ingestion
box(45, 52, 17, 16, "Event streaming log\n(pub/sub, partitioned by\nmeter / substation)\n• at-least-once delivery\n• 7-day replay buffer", "ing", fs=6.9)
box(45, 38, 17, 11, "Schema registry +\ndata contracts\n(versioned per vendor)", "ing", fs=6.9)
box(45, 23, 17, 12, "Batch landing zone\n(files arrive as\nbounded streams)", "ing", fs=6.9)
# storage
box(68, 61, 19, 7.5, "Bronze - raw, immutable\n(object storage, replayable)", "sto", fs=6.9)
box(68, 51.5, 19, 7.5, "Silver - curated, UTC,\ndeduplicated, flagged", "sto", fs=6.9)
box(68, 42, 19, 7.5, "Gold - serving aggregates\n& feature tables", "sto", fs=6.9)
box(68, 32.5, 19, 7.5, "Quarantine - rejected rows\nwith reason codes", "sto", fs=6.9)
box(68, 19.5, 19, 10.5, "Operational DB (PII):\ncustomer, contract, billing\nseparate keys & retention", "sto", fs=6.9)
# processing
box(93, 54, 18, 14, "Unified Spark engine\n• Structured Streaming\n  (seconds latency)\n• replay / backfill on\n  same code (Kappa)", "pro", fs=6.9)
box(93, 40, 18, 11, "Data-quality rules\n& observability metrics\n(lag, rejects, freshness)", "pro", fs=6.9)
box(93, 26.5, 18, 11.5, "ML platform\n• peak-demand forecast\n• voltage anomaly scoring\n• model registry,\n  drift monitoring", "pro", fs=6.9)
# consumption
box(117, 59, 15.5, 9, "Control-room\ndashboards & alerts", "con", fs=6.9)
box(117, 47.5, 15.5, 9, "Demand-response &\nnetwork planning", "con", fs=6.9)
box(117, 36, 15.5, 9, "Regulatory reporting\n(Ofgem / NRAs, GDPR)", "con", fs=6.9)
box(117, 24.5, 15.5, 9, "Billing & customer\napps (data API)", "con", fs=6.9)

# ---- flows
for y in [65.6, 59, 52.4, 45.8]:
    arrow(17.9, y, 22.9, 62 if y > 55 else 47)
for y in [32.6, 26]:
    arrow(17.9, y, 22.9, 32)
arrow(17.9, 39.2, 22.9, 32)
arrow(39.2, 62, 44.9, 61); arrow(39.2, 47, 44.9, 58); arrow(39.2, 32, 44.9, 29)
arrow(53.5, 51.9, 53.5, 49.1, col=LAYER["ing"][1]); arrow(53.5, 37.9, 53.5, 35.1, col=LAYER["ing"][1])
arrow(62.1, 60, 67.9, 64.5); arrow(62.1, 29, 67.9, 63)
arrow(77.5, 60.9, 77.5, 59.1, col=LAYER["sto"][1]); arrow(77.5, 51.4, 77.5, 49.6, col=LAYER["sto"][1])
arrow(87.1, 64.5, 92.9, 61); arrow(92.9, 57, 87.1, 55, col=LAYER["pro"][1]); arrow(92.9, 45, 87.1, 36, col=LAYER["pro"][1])
arrow(87.1, 45, 92.9, 34)
arrow(111.1, 61, 116.9, 63.5); arrow(111.1, 33, 116.9, 52); arrow(87.1, 44, 116.9, 40)
arrow(87.1, 22.5, 116.9, 27)
arrow(39.2, 59, 44.9, 66, col=LAYER["edge"][1], ls="--")

# ---- cross-cutting bands
ax.add_patch(FancyBboxPatch((1, 9.2), 133, 5.6, boxstyle="round,pad=0.25,rounding_size=1.2", fc="#fdecec", ec="#e34948", lw=1.4))
ax.text(67.5, 12, "SECURITY & GOVERNANCE (every layer):  IAM / least privilege  •  encryption in transit (TLS) & at rest (customer-managed keys)\n"
        "data catalogue & lineage  •  pseudonymisation of meter IDs  •  audit logs  •  DPIA  •  NIS2 / ISO 27001 / 27019 controls",
        ha="center", va="center", fontsize=7.9, color=INK)
ax.add_patch(FancyBboxPatch((1, 1.6), 133, 5.6, boxstyle="round,pad=0.25,rounding_size=1.2", fc="#eef4ea", ec="#008300", lw=1.4))
ax.text(67.5, 4.4, "OPERATIONS (every layer):  workflow orchestration  •  infrastructure-as-code  •  SLO monitoring (ingestion lag, reject rate, freshness)\n"
        "cost tags & autoscaling  •  UK region + EU region, metadata replicated, data resident  •  disaster recovery tested quarterly",
        ha="center", va="center", fontsize=7.9, color=INK)
ax.text(116, 18.2, "dashed = edge alerts bypass cloud", fontsize=6.8, color=INK2, ha="left")

fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("saved", OUT)

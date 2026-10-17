from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = "report/figures"


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def make_all(d):
    # EDA yalnız train hissəsində edilir ki, test datasına baxılmasın
    df = d["data"].iloc[d["train"]]
    print(f"\n[eda] train rows: {len(df)}")
    print(df[["price", "area", "rooms", "floor", "total_floors", "land_area"]].describe().round(1).to_string())
    print("\ncategory counts:")
    print(df["category"].fillna("unknown").value_counts().to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(df["price"] / 1000, bins=80, color="#1f5fad")
    axes[0].set_xlabel("price (thousand AZN)")
    axes[0].set_ylabel("listings")
    axes[0].set_title("Price is heavy-tailed")
    axes[1].hist(np.log(df["price"]), bins=80, color="#c4511f")
    axes[1].axvline(np.log(d["thr"]), color="k", ls="--", lw=1, label=f"tier threshold {d['thr']:,.0f} AZN")
    axes[1].set_xlabel("log(price)")
    axes[1].set_title("log(price) is close to symmetric")
    axes[1].legend(fontsize=8)
    save(fig, "fig_price_hist.png")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    has_area = df["area"].notna()
    for cat in df.loc[has_area, "category"].fillna("unknown").value_counts().index[:5]:
        sel = has_area & (df["category"].fillna("unknown") == cat)
        ax.scatter(df.loc[sel, "area"], df.loc[sel, "price"], s=3, alpha=0.3, label=cat)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("area (m²)")
    ax.set_ylabel("price (AZN)")
    ax.set_title("Area vs price (log-log)")
    ax.legend(markerscale=4, fontsize=7)
    save(fig, "fig_area_vs_price.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    rooms = df["rooms"].clip(upper=6)
    groups = [np.log(df.loc[rooms == r, "price"]) for r in range(1, 7)]
    ax.boxplot(groups, showfliers=False)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(["1", "2", "3", "4", "5", "6+"])
    ax.set_xlabel("rooms")
    ax.set_ylabel("log(price)")
    ax.set_title("Price by number of rooms")
    save(fig, "fig_rooms_box.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    geo = df[df["lat"].notna() & df["lng"].notna()]
    lat_lo, lat_hi = geo["lat"].quantile([0.01, 0.99])
    lng_lo, lng_hi = geo["lng"].quantile([0.01, 0.99])
    geo = geo[geo["lat"].between(lat_lo, lat_hi) & geo["lng"].between(lng_lo, lng_hi)]
    sc = ax.scatter(geo["lng"], geo["lat"], c=np.log(geo["price"]), s=2, cmap="viridis", alpha=0.6)
    fig.colorbar(sc, ax=ax, label="log(price)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Location effect (central 98% of coordinates)")
    save(fig, "fig_map_latlng.png")

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    parts = [("train", d["t_tr"]), ("validation", d["t_val"]), ("test", d["t_te"])]
    x = np.arange(len(parts))
    std = [int((t == 0).sum()) for _, t in parts]
    prem = [int((t == 1).sum()) for _, t in parts]
    ax.bar(x - 0.2, std, 0.4, label="standard (0)", color="#1f5fad")
    ax.bar(x + 0.2, prem, 0.4, label="premium (1)", color="#c4511f")
    ax.set_xticks(x)
    ax.set_xticklabels([p for p, _ in parts])
    ax.set_ylabel("listings")
    ax.set_title("Price-tier class balance")
    ax.legend(fontsize=8)
    save(fig, "fig_tier_balance.png")
    for (name, t), s, p in zip(parts, std, prem):
        print(f"  {name:<10} standard={s}  premium={p}  premium share={p / (s + p):.3f}")

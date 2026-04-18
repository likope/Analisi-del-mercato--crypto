import requests
import polars as pl
from scipy.stats import norm
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import norm as sp_norm
import math
import matplotlib.pyplot as plt

# 1. Prendi tutti gli strumenti ETH opzione
instruments = requests.get(
    "https://www.deribit.com/api/v2/public/get_instruments",
    params={"currency": "ETH", "kind": "option", "expired": "false"}
).json()["result"]

# 2. Filtra per la scadenza che ti interessa
expiry = "24APR26"
filtered = [i for i in instruments if expiry in i["instrument_name"]]

# 3. Per ogni strumento, prendi il ticker
rows = []
for inst in filtered:
    name = inst["instrument_name"]
    ticker = requests.get(
        "https://www.deribit.com/api/v2/public/ticker",
        params={"instrument_name": name}
    ).json()["result"]
    
    rows.append({
        "instrument": name,
        "strike": inst["strike"],
        "type": inst["option_type"],        # "call" o "put"
        "gamma": ticker["greeks"]["gamma"],
        "delta": ticker["greeks"]["delta"],
        "vega": ticker["greeks"]["vega"],
        "iv": ticker["mark_iv"],
        "oi": ticker["open_interest"],
        "volume": ticker["stats"]["volume"],
        "bid": ticker["best_bid_price"],
        "ask": ticker["best_ask_price"],
        "mark": ticker["mark_price"],
    })

    spot = requests.get(
    "https://www.deribit.com/api/v2/public/get_index_price",
    params={"index_name": "eth_usd"}
    ).json()["result"]["index_price"]

# 4. DataFrame pronto
df = pl.DataFrame(rows)
print(df.head)

def netgex(df, spot):
    """
    Calcolo del NETGEX per ogni contratto, formula: GEXi​=OIi​×γi​×S2×0.01
    """

    df = df.with_columns(
        pl.when(pl.col("type") == "call")
        .then(pl.lit(1))
        .otherwise(pl.lit(-1))
        .alias("sign")
    ).with_columns(
        (pl.col("sign")*pl.col("oi")*pl.col("gamma")*spot**2*0.01).alias("gex"))
    return df

def agg_gex(df):
    df = (
    df.group_by("strike")
      .agg([
          # Net GEX totale per strike
          pl.col("gex").sum().alias("net_gex"),

          # GEX separato call e put (per identificare walls)
          pl.col("gex").filter(pl.col("type") == "call").sum().alias("gex_call"),
          pl.col("gex").filter(pl.col("type") == "put").sum().alias("gex_put"),

          # OI separato (per contesto)
          pl.col("oi").filter(pl.col("type") == "call").sum().alias("oi_call"),
          pl.col("oi").filter(pl.col("type") == "put").sum().alias("oi_put"),

          # Gamma-weighted OI (per identificare walls)
          (pl.col("oi") * pl.col("gamma"))
              .filter(pl.col("type") == "call").sum()
              .alias("weight_call"),
          (pl.col("oi") * pl.col("gamma"))
              .filter(pl.col("type") == "put").sum()
              .alias("weight_put"),
      ])
      .sort("strike")
    )
    return df

def gamma_flip(df, spot, expiry_str):
    """
    Scanna prezzi ipotetici, ricalcola gamma BS a ogni S,
    trova dove Net GEX cambia segno.
    """
    # TTE in anni
    from datetime import datetime, timezone
    exp_dt = datetime.strptime(expiry_str, "%d%b%y").replace(
        tzinfo=timezone.utc, hour=8
    )
    now = datetime.now(timezone.utc)
    tte = (exp_dt - now).total_seconds() / (365.25 * 24 * 3600)
    if tte <= 0:
        print("  Scadenza passata")
        return None

    # Dati necessari come numpy arrays
    K = df["strike"].to_numpy()
    iv = df["iv"].to_numpy() / 100       # Deribit dà IV in %
    oi = df["oi"].to_numpy()
    is_call = (df["type"] == "call").to_numpy()

    # Griglia di prezzi ipotetici: ±20% da spot
    prices = np.linspace(spot * 0.80, spot * 1.20, 300)
    net_gex = np.zeros_like(prices)

    for i, S in enumerate(prices):
        # Gamma BS ricalcolato a questo S ipotetico
        valid = (iv > 0) & (K > 0)
        d1 = np.zeros_like(K)
        d1[valid] = (
            (np.log(S / K[valid]) + 0.5 * iv[valid]**2 * tte)
            / (iv[valid] * np.sqrt(tte))
        )
        gamma = np.zeros_like(K)
        gamma[valid] = norm.pdf(d1[valid]) / (S * iv[valid] * np.sqrt(tte))

        # GEX: call +, put -
        contrib = oi * gamma * S**2 * 0.01
        net_gex[i] = contrib[is_call].sum() - contrib[~is_call].sum()

    # Trova zero crossing più vicino a spot
    sign_changes = np.where(np.diff(np.sign(net_gex)) != 0)[0]
    if len(sign_changes) == 0:
        print("  Nessun gamma flip trovato nel range")
        return None

    candidates = (prices[sign_changes] + prices[sign_changes + 1]) / 2
    flip = float(candidates[np.argmin(np.abs(candidates - spot))])
    return flip

def bs_call_price(S, K, T, sigma, r=0.0):
    """Prezzo call Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * sp_norm.cdf(d1) - K * math.exp(-r * T) * sp_norm.cdf(d2)


def risk_neutral_density(df_raw, spot, expiry_str, r=0.0):
    """
    Breeden-Litzenberger: q(K) = e^(rT) · ∂²C/∂K²

    Strategia:
      1) Prendi solo OTM (più liquide):
         - strike < spot: usa put, converti in call via parity
         - strike >= spot: usa call direttamente
      2) Cubic spline sulla IV (non sui prezzi — meno rumore)
      3) Ricalcola call prices su griglia fine dalla IV smoothed
      4) Derivata seconda numerica → densità
      5) Clip >= 0 (no arbitraggio), normalizza a integrale 1
    """
    # TTE
    from datetime import datetime, timezone
    exp_dt = datetime.strptime(expiry_str, "%d%b%y").replace(
        tzinfo=timezone.utc, hour=8
    )
    now = datetime.now(timezone.utc)
    T = (exp_dt - now).total_seconds() / (365.25 * 24 * 3600)
    if T <= 0:
        print("  Scadenza passata")
        return None

    # Separa call e put, filtra OI > 0 e IV > 0
    calls = (df_raw.filter(
                (pl.col("type") == "call") &
                (pl.col("oi") > 0) &
                (pl.col("iv") > 0)
             ).sort("strike"))

    puts = (df_raw.filter(
                (pl.col("type") == "put") &
                (pl.col("oi") > 0) &
                (pl.col("iv") > 0)
            ).sort("strike"))

    # OTM put (strike < spot): converti in call via put-call parity
    # C = P + S - K·e^(-rT)
    otm_puts = (
        puts.filter(pl.col("strike") < spot)
            .with_columns(
                # mark è in unità di sottostante, converti in USD
                (pl.col("mark") * spot + spot - pl.col("strike") * math.exp(-r * T))
                    .alias("call_usd"),
                (pl.col("iv") / 100).alias("iv_dec"),
            )
            .select(["strike", "call_usd", "iv_dec"])
    )

    # OTM call (strike >= spot): prezzo diretto
    otm_calls = (
        calls.filter(pl.col("strike") >= spot)
             .with_columns(
                 (pl.col("mark") * spot).alias("call_usd"),
                 (pl.col("iv") / 100).alias("iv_dec"),
             )
             .select(["strike", "call_usd", "iv_dec"])
    )

    # Combina, ordina, dedupl
    combined = (
        pl.concat([otm_puts, otm_calls])
          .sort("strike")
          .unique(subset=["strike"], keep="first")
    )

    if len(combined) < 6:
        print("  Troppi pochi strike per densità robusta")
        return None

    strikes = combined["strike"].to_numpy()
    ivs = combined["iv_dec"].to_numpy()

    # Spline su IV → smooth, poi ricalcola prezzi call su griglia fine
    cs = CubicSpline(strikes, ivs, extrapolate=False)
    K_grid = np.linspace(strikes[0], strikes[-1], 500)
    iv_grid = np.clip(cs(K_grid), 1e-4, 5.0)

    # Prezzi call BS sulla griglia fine con IV smoothed
    C_grid = np.array([
        bs_call_price(spot, K, T, s, r)
        for K, s in zip(K_grid, iv_grid)
    ])

    # Derivata seconda numerica: ∂²C/∂K²
    dK = K_grid[1] - K_grid[0]
    d2C = np.gradient(np.gradient(C_grid, dK), dK)

    # Densità: q(K) = e^(rT) · ∂²C/∂K²
    density = np.clip(math.exp(r * T) * d2C, 0, None)

    # Normalizza a integrale 1
    total = np.trapezoid(density, K_grid)
    if total > 0:
        density = density / total

    return K_grid, density


def plot_density(K_grid, density, spot, flip, walls_call, walls_put):
    """
    Grafico della densità risk-neutral con livelli operativi.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Densità
    ax.fill_between(K_grid, density, alpha=0.3, color="steelblue")
    ax.plot(K_grid, density, color="steelblue", linewidth=2, label="Densità Q")

    # Spot
    ax.axvline(spot, color="white", linewidth=2, linestyle="-", label=f"Spot ${spot:,.0f}")

    # Gamma flip
    if flip:
        ax.axvline(flip, color="yellow", linewidth=1.5, linestyle="--",
                   label=f"Gamma Flip ${flip:,.0f}")

    # Call walls (rosso)
    for row in walls_call.iter_rows(named=True):
        ax.axvline(row["strike"], color="red", linewidth=1, linestyle=":",
                   alpha=0.8)
        ax.text(row["strike"], max(density) * 0.95, f"CW {row['strike']:,.0f}",
                color="red", fontsize=8, ha="center", rotation=90)

    # Put walls (verde)
    for row in walls_put.iter_rows(named=True):
        ax.axvline(row["strike"], color="lime", linewidth=1, linestyle=":",
                   alpha=0.8)
        ax.text(row["strike"], max(density) * 0.95, f"PW {row['strike']:,.0f}",
                color="lime", fontsize=8, ha="center", rotation=90)

    # Quantili shading (25-75)
    cdf = np.concatenate([[0], np.cumsum(
        (density[:-1] + density[1:]) / 2 * np.diff(K_grid)
    )])
    cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf
    q25 = float(np.interp(0.25, cdf, K_grid))
    q75 = float(np.interp(0.75, cdf, K_grid))
    mask = (K_grid >= q25) & (K_grid <= q75)
    ax.fill_between(K_grid[mask], density[mask], alpha=0.2, color="orange",
                    label=f"IQR ${q25:,.0f}–${q75:,.0f}")

    ax.set_xlabel("Strike ($)", fontsize=12)
    ax.set_ylabel("Densità", fontsize=12)
    ax.set_title("Risk-Neutral Density (Breeden-Litzenberger)", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)

    # Zoom sul range rilevante
    ax.set_xlim(spot * 0.88, spot * 1.12)

    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.legend(facecolor="#1a1a2e", edgecolor="gray", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("gray")

    plt.tight_layout()
    plt.show()


def density_report(K_grid, density, spot):
    """Calcola e stampa momenti della distribuzione risk-neutral."""

    q = density

    # Momenti
    mean = np.trapz(K_grid * q, K_grid)
    var  = np.trapz((K_grid - mean)**2 * q, K_grid)
    std  = np.sqrt(max(var, 0))
    skew = np.trapz(((K_grid - mean) / std)**3 * q, K_grid) if std > 0 else 0
    kurt = np.trapz(((K_grid - mean) / std)**4 * q, K_grid) - 3 if std > 0 else 0
    mode = float(K_grid[np.argmax(q)])

    # CDF per quantili
    cdf = np.concatenate([[0], np.cumsum((q[:-1] + q[1:]) / 2 * np.diff(K_grid))])
    cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf

    def quantile(p):
        return float(np.interp(p, cdf, K_grid))

    q05, q25, q50, q75, q95 = [quantile(p) for p in [0.05, 0.25, 0.50, 0.75, 0.95]]
    # Stampa
    print(f"\n📊 RISK-NEUTRAL DENSITY")
    print(f"  Mean (Q):    ${mean:,.2f}")
    print(f"  Median:      ${q50:,.2f}")
    print(f"  Mode:        ${mode:,.2f}")
    print(f"  Std:         ${std:,.2f}  ({std/spot*100:.2f}% di spot)")

    print(f"\n  Quantili:")
    print(f"    5%:   ${q05:,.0f}")
    print(f"    25%:  ${q25:,.0f}")
    print(f"    75%:  ${q75:,.0f}")
    print(f"    95%:  ${q95:,.0f}")

    skew_desc = ("left-tail fear" if skew < -0.2
                 else "right-tail greed" if skew > 0.2
                 else "quasi-simmetrica")
    kurt_desc = ("fat tails" if kurt > 1
                 else "thin tails" if kurt < -0.5
                 else "normale-ish")
    print(f"\n  Skew:        {skew:+.3f}  ({skew_desc})")
    print(f"  Excess kurt: {kurt:+.3f}  ({kurt_desc})")

    # Drift check
    drift_pct = (mean - spot) / spot * 100
    print(f"\n  Drift (mean-spot): {drift_pct:+.2f}%")
    if abs(drift_pct) > 1:
        print(f"    ⚠ Drift > 1% senza r: check liquidity/parity")

    # Probabilità implicite utili per trading
    prob_below_spot = quantile_to_prob(cdf, K_grid, spot)
    print(f"\n  P(ETH < spot a expiry):  {prob_below_spot*100:.1f}%")
    print(f"  P(ETH > spot a expiry):  {(1-prob_below_spot)*100:.1f}%")


def quantile_to_prob(cdf, K_grid, level):
    """Probabilità implicita che il prezzo sia sotto un certo livello."""
    return float(np.interp(level, K_grid, cdf))

df = netgex(df, spot)
df_copy = df
df = agg_gex(df)
print(df.head())
net_gex_total = df["net_gex"].sum()
regime = "LONG GAMMA (pin)" if net_gex_total > 0 else "SHORT GAMMA (amplifica)"
print(f"Spot= {spot}")
print(f"Net GEX totale= {net_gex_total}")

call_wall = (
    df.filter(pl.col("strike") > spot)
           .sort("weight_call", descending=True)
           .head(3)
)
put_wall = (
    df.filter(pl.col("strike") < spot)
           .sort("weight_put", descending=True)
           .head(3)
)

print("\nCALL WALLS (resistenza dealer):")
for row in call_wall.iter_rows(named=True):
    print(f"  ${row['strike']:>8,.0f}  "
          f"OI×Γ: {row['weight_call']:>8.4f}  "
          f"GEX: ${row['gex_call']:>12,.0f}  "
          f"OI: {row['oi_call']:>6,}")

print("\nPUT WALLS (supporto dealer):")
for row in put_wall.iter_rows(named=True):
    print(f"  ${row['strike']:>8,.0f}  "
          f"OI×Γ: {row['weight_put']:>8.4f}  "
          f"GEX: ${row['gex_put']:>12,.0f}  "
          f"OI: {row['oi_put']:>6,}")
    
flip = gamma_flip(df_copy, spot, expiry)

print(f"\n⚡ GAMMA FLIP: ${flip:,.2f}")
if flip > spot:
    diff = (flip - spot) / spot * 100
    print(f"  Flip SOPRA spot di {diff:.2f}%")
    print(f"  → Spot in zona SHORT GAMMA: dealer amplificano")
    print(f"  → SL per long: sopra il flip non serve, sotto il flip accelera")
else:
    diff = (spot - flip) / spot * 100
    print(f"  Flip SOTTO spot di {diff:.2f}%")
    print(f"  → Spot in zona LONG GAMMA: dealer smorzano (pin)")
    print(f"  → SL per long: sotto ${flip:,.0f} (regime cambia)")

result = risk_neutral_density(df_copy, spot, expiry)
if result is not None:
    K_grid, density = result
    density_report(K_grid, density, spot)
    plot_density(K_grid, density, spot, flip, call_wall, put_wall)
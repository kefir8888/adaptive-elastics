# Cheaper GPU cloud + crypto payment from Russia

> **⚠ SUPERSEDED (2026-06-16) — DO NOT use this as the live plan.** The decision is to stay on
> **immers.cloud, paid in rubles**. Crypto-funded Western providers (**Vast.ai / RunPod /
> Spheron**) are **dropped**: their "crypto" routes through KYC + sanctions-screening processors
> (BitPay / Crypto.com) that **block Russian users**, and the ~$5–10 saving is not worth the
> hassle for a **~$18–20** experiment. **Revisit crypto only** if a large final batch of
> confirmation seeds is ever needed. The body below is kept as a **historical price/payment
> reference** and is **stale** — the current infrastructure plan is in `docs/NEXT_SESSION.md`.

**Purpose.** We currently rent an immers.cloud H100 PCIe at **342 RUB/hr**. This
document (a) compares cheaper Western GPU clouds for H100 / A100 / RTX-4090-class
training and (b), since sanctions block most Russian cards, gives a step-by-step
recipe to fund those accounts with **crypto (USDT-TRC20)** from Russia.

> Written 2026-06-16. Cloud prices are a **live marketplace** for Vast.ai / RunPod
> community / TensorDock and move daily; treat the numbers as a snapshot, not a quote.

## Assumptions

- **FX rate: 1 USD = 72.5 RUB** — the spot mid-rate on 2026-06-14
  ([exchange-rates.org](https://www.exchange-rates.org/exchange-rate-history/usd-rub-2026)).
  The ruble strengthened ~8 % YTD in 2026; the 2026 *average* was ~77 RUB/USD, and the
  range was 71–86. I use **72.5** for the headline and note that at the more conservative
  77 the dollar prices below cost ~6 % more rubles.
  - *Sanity check on the baseline:* 342 RUB/hr ÷ 72.5 = **$4.72/hr**. (The brief's
    "~$3.7/hr" implies ~92 RUB/USD, a 2024–25-era rate; at today's stronger ruble the
    immers.cloud rate is effectively **~$4.7/hr** in dollar terms. I compare against the
    fixed **342 RUB/hr** so the FX assumption only shifts the competitors, not the baseline.)
- **Workload = single-GPU PPO** (MJX, ~47k env-steps/s, ~70 min per 200M-step run, per
  CLAUDE.md). One GPU at a time; no multi-node InfiniBand needed. This means the cheap
  **marketplace / community** tiers are fair game — we do not need enterprise SLAs.
- **On-demand**, not spot/interruptible, for the headline (a 70-min run that gets
  preempted at minute 60 wastes money). Spot prices are listed as the cheaper floor for
  the cost-tolerant; with checkpoint-resume (the project already has Colab resume) spot
  becomes attractive.
- An H100 is **not required** — the run is short and a 4090/A100 finishes a 200M-step
  job in a few hours for a fraction of the price. The 4090 tier is where the real savings
  are.

## Provider price comparison (vs 342 RUB/hr)

Per-GPU on-demand USD/hr, converted at 72.5 RUB/USD. "RUB/hr" is the directly comparable
column; "x cheaper" = 342 / (provider RUB/hr). Marketplace lows (Vast/RunPod-community/
TensorDock) are the cheapest *available* host, not a guaranteed rate.

| Provider | GPU | USD/hr (on-dem.) | ≈ RUB/hr | vs 342 RUB/hr | Crypto pay? |
|---|---|---|---|---|---|
| **immers.cloud (baseline)** | H100 PCIe | ~$4.72 | **342** | — | (RU card) |
| Spheron (DePIN) | H100 SXM5 | $2.50 (PCIe $2.01) | 181 / 146 | **1.9–2.3x** | **Yes — USDT/USDC** |
| FluidStack | H100 SXM | $2.10 | 152 | **2.2x** | No (card/invoice) |
| TensorDock | H100 SXM5 | $2.25 | 163 | **2.1x** | Unconfirmed |
| Hyperstack | H100 PCIe | $1.90 | 138 | **2.5x** | No (card/invoice) |
| RunPod (Secure) | H100 PCIe | $2.39–2.89 | 173–210 | **1.6–2.0x** | **Yes — crypto** |
| Vast.ai (marketplace) | H100 | $1.49–2.27 | 108–165 | **2.1–3.2x** | **Yes — Crypto.com/BitPay** |
| Lambda | H100 PCIe | $3.29 | 238 | **1.4x** | No (card only) |
| **A100 tier** | | | | | |
| Spheron | A100 80GB | $1.07 (spot $0.60) | 78 (44) | **4.4x (7.8x)** | **Yes** |
| Thunder Compute | A100 | $0.78 | 57 | **6.0x** | No |
| TensorDock | A100 | ~$1.30 | 94 | **3.6x** | Unconfirmed |
| RunPod | A100 PCIe | $1.39 | 101 | **3.4x** | **Yes** |
| Vast.ai | A100 80GB | $0.67–1.10 | 49–80 | **4.3–7.0x** | **Yes** |
| **RTX 4090 tier** (best value) | | | | | |
| TensorDock | RTX 4090 | ~$0.25 | 18 | **19x** | Unconfirmed |
| Vast.ai | RTX 4090 | $0.31–0.55 | 22–40 | **8.5–15x** | **Yes** |
| RunPod (Community) | RTX 4090 | $0.34–0.69 | 25–50 | **6.8–14x** | **Yes** |
| Spheron | RTX 4090 | $0.55 | 40 | **8.6x** | **Yes** |

Sources:
[Spheron](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/),
[RunPod pricing](https://www.runpod.io/product/cloud-gpus),
[Vast.ai](https://vast.ai/),
[Lambda](https://www.spheron.network/blog/lambda-cloud-h100-pricing-2026/),
[TensorDock H100](https://www.tensordock.com/gpu-h100.html),
[Hyperstack H100](https://www.hyperstack.cloud/h100-pcie),
[Northflank cheapest list](https://northflank.com/blog/cheapest-cloud-gpu-providers),
[IntuitionLabs H100 survey](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison).

**Reading the table:** every credible Western provider beats 342 RUB/hr on H100
(1.4–3.2x cheaper), and the savings *explode* if we drop to a 4090 (6–19x). For our
short PPO runs the **RTX 4090 is the rational default** — a 200M-step job that costs
~342 RUB/hr × ~1.2 h ≈ 410 RUB on the H100 baseline costs **~25–50 RUB** on a 4090.

## Cheapest viable options

Filtered to *crypto-payable* (mandatory from Russia) and *reliable enough for a 1–3 h run*:

1. **Vast.ai — top pick.** Cheapest marketplace floor in every tier, mature crypto
   funding (Crypto.com / BitPay), per-second billing, one-GPU rentals trivial. Filter
   hosts by reliability/DLPerf and pick a verified datacenter host, not a random
   residential one. **4090 @ ~22–40 RUB/hr (≈8–15x cheaper); H100 @ ~108–165 RUB/hr.**
2. **RunPod — most ergonomic.** Slightly pricier than Vast but fixed templates, good
   PyTorch/MJX images, reliable Secure Cloud, accepts crypto. Use **Community Cloud**
   4090 (~25–50 RUB/hr) for cost or Secure H100 (~173–210 RUB/hr) for reliability.
3. **Spheron (DePIN) — H100 value + native stablecoins.** Pays directly in **USDT/USDC**
   (no card ever, ideal for Russia), per-minute billing, H100 SXM5 from $2.50/hr
   (≈181 RUB/hr, ~1.9x) and A100 80GB from $1.07 (≈78 RUB/hr, ~4.4x). Best if you want
   an H100 specifically and want to skip card processors entirely.

**Avoid for our case:** Lambda (card-only, no crypto, only 1.4x cheaper), the
hyperscalers (AWS/Azure/GCP — *more* expensive than baseline and card-only), and
TensorDock/Hyperstack/Thunder/FluidStack **only because their crypto support is
unconfirmed** — they are price-competitive but verify the payment path before relying
on them from Russia.

> **Net recommendation:** default to **Vast.ai RTX 4090** for routine PPO runs
> (~8–15x cheaper, crypto OK); use **Spheron or Vast H100** when you actually need
> H100-class throughput. Keep ~$20–50 of USDT-TRC20 pre-loaded so a run never stalls
> on funding.

## Crypto payment step-by-step from Russia

Goal: turn rubles into **USDT on the TRON network (TRC-20)** — the de-facto standard in
Russia: cheap (~1 USDT) and fast (1–3 min) transfers, and what most providers/processors
accept — then load it onto the GPU provider.

### Step 1 — Acquire USDT-TRC20 with rubles (P2P)

The reliable 2026 route is a centralized exchange's **P2P desk** (escrow-protected),
funded by a ruble card transfer. Common venues for Russian users: **Bybit, OKX, KuCoin,
HTX** ([review](https://moscow-city.guide/en/kriptovlyuta/bybit-v-rossii-v-2026-godu-kak-rabotaet-birzha/)).

1. **Register** on Bybit or OKX (email/phone + password). Pass **KYC** (passport photo +
   Face ID) — needed to lift P2P limits.
2. Open **P2P → Buy → USDT**, set fiat = **RUB**, and filter by **payment method =
   your Russian bank** (e.g. a bank you actually hold a card with). Sort by price.
3. Pick a **seller with high completion rate + many trades**, within your amount range.
   Open the order; the seller's USDT is now **locked in escrow**.
4. **Transfer rubles** from your bank to the seller's stated account *exactly as shown*
   (right amount, right account, no crypto words in the payment comment).
5. **Only after the money leaves your bank**, click "I have paid." The seller releases
   USDT from escrow to your exchange account. **Never** confirm before paying, and
   **never** take the deal off-platform.
6. Expect to pay **~0.5–2 % over mid-market**; there is usually **no buyer-side platform
   fee** on Bybit/OKX P2P.

### Step 2 — Withdraw USDT on the TRC-20 network

1. In the exchange, **Withdraw → USDT → network = TRON (TRC-20)**.
2. Paste your destination address — either the **provider's deposit address** (if it
   gives one directly, e.g. Spheron/Akash-style stablecoin top-up) or **your own
   non-custodial wallet** (Trust Wallet / Tronlink) if you want to hold/route it.
3. Network fee is small (a few TRX, ≈1 USDT-equivalent). Keep a little **TRX** in any
   self-custody wallet to pay for the *send*. **Send a small test amount first** for any
   new address.

### Step 3 — Fund the GPU provider

- **Spheron / Akash (DePIN):** simplest — pay the platform **directly in USDT/USDC**.
  Top up the escrow/balance from your wallet; no card, no processor. Best Russia fit.
- **Vast.ai:** Add Credit → choose the **crypto option (Crypto.com Pay / BitPay)**. The
  processor shows a deposit address/QR; send USDT (use the **network the processor
  asks for** — TRC-20 if offered, else ERC-20). Note: Vast also wants a card on file
  for some flows, but balance top-ups can be crypto.
- **RunPod:** Billing → add funds → **crypto** path (Crypto.com). Same pattern: send to
  the shown address, balance credits after confirmations. Card is *not* required for the
  crypto top-up; keep a USDT buffer so pods never pause mid-run.
- After funding, the balance is in **USD credit** — billing then proceeds per-second/
  per-minute exactly like a card account.

### Practical end-to-end

`RUB card → Bybit/OKX P2P → USDT (exchange) → withdraw TRC-20 → provider deposit
(Spheron direct, or Crypto.com/BitPay for Vast/RunPod) → USD credit → rent GPU.`
Budget ~10–20 min the first time; subsequent top-ups are ~2 min.

## Risks / caveats

- **FX uncertainty.** Headline uses 72.5 RUB/USD; at the 2026-average 77 the dollar
  prices are ~6 % more rubles. Every competitor still beats 342 RUB/hr by the same
  factor minus that ~6 %. The 4090 verdict (6–19x) is robust to any plausible FX.
- **Marketplace prices are not quotes.** Vast/RunPod-community/TensorDock lows depend on
  *which host is free right now*. Check live before assuming the cheap rate; reliability
  varies host-to-host (filter on reliability score, prefer datacenter over residential).
- **161-FZ / 115-FZ account-freeze risk (the big Russia gotcha).** Since May 2025,
  tightened controls on P2P ruble transfers mean some Russian recipients/senders see
  **bank accounts blocked** after P2P crypto payments
  ([source](https://moscow-city.guide/en/articles/gde-kupit-usdt-v-moskve-v-2025-godu-sravnenie-populyarnykh-sposobov-i-sovety-po-pokupke/)).
  Mitigations: use a **reputable high-volume seller**, keep amounts modest, **never**
  write "crypto/USDT/Bybit" in the bank transfer comment, avoid many rapid P2P transfers,
  and consider a secondary bank card for this purpose.
- **Scams on P2P.** Only release/confirm via the platform escrow; verify funds actually
  arrived; never move off-platform; double-check wallet addresses (address-swap malware
  exists).
- **Wrong-network loss.** Sending USDT on the wrong chain (ERC-20 address vs TRC-20
  send, or vice-versa) can be **unrecoverable**. Match the network on both ends; test
  with a small amount.
- **Sanctions / KYC on the exchange.** A Western processor (Crypto.com/BitPay/Stripe-on-
  crypto) may apply geo/KYC checks. Funding via stablecoin to a DePIN provider (Spheron/
  Akash) sidesteps card processors entirely and is the most robust Russia path — prefer
  it when reliability allows.
- **CEX custodial risk.** Don't park large balances on the exchange; withdraw promptly to
  self-custody or straight to the provider. Only convert what a few runs need.
- **No-regen ≠ no-refund, but no refunds either.** Crypto top-ups are generally
  non-refundable; size deposits to expected usage (a 4090 PPO run is cents, so small
  top-ups suffice).
- **Provider crypto support can change.** "Yes" entries verified mid-2026 via docs/blog;
  re-confirm the live billing page before a large top-up. TensorDock/Hyperstack crypto
  is **unconfirmed** here — verify before committing.

---
*Sources consolidated:* Spheron, RunPod, Vast.ai, Lambda, TensorDock, Hyperstack,
Northflank, IntuitionLabs (prices); Vast.ai billing docs, RunPod funding blog,
moscow-city.guide Bybit/USDT-Moscow guides, exchange-rates.org (FX) — URLs inline above.

---
## DECISION + CORRECTION (2026-06-16)
**Correction to the earlier sections:** the mainstream providers' "crypto" is NOT a direct USDT deposit —
**Vast.ai** routes through **BitPay / Crypto.com** and RunPod through similar processors, all of which KYC +
sanctions-screen and likely **block Russian users**. The earlier "just send USDT-TRC20" guidance does **not**
apply to them.

**Chosen provider: Spheron** (managed GPU cloud, `app.spheron.ai` / `docs.spheron.ai` — NOT the decentralized
Protocol). Rationale:
- **USDT/USDC accepted directly** → works from Russia, no KYC processor.
- **API + CLI** for programmatic instance lifecycle → the agent can create/destroy boxes (no manual SSH-paste).
- **Bare-metal + SSH + dedicated IP**, Tier 3/4 datacenters, 99.9% SLA → `gpu_box_setup.sh` runs unchanged.
- **A100 ~$0.85/hr** (a ~25–30 min Go1 run ≈ **$0.35–0.45**). H100 ~$1.46, 4090 ~$0.53.
- Clore.ai is ~15–30% cheaper but pays in CLORE/BTC (token conversion) and is a variable marketplace.

**Setup (user does 1–3, then hands the agent the API key):**
1. Create an account at `app.spheron.ai`.
2. Fund with **USDT** (deposit the P2P-acquired stablecoin).
3. Generate an **API key** (dashboard → settings/API).
4. Hand the API key to the agent → it installs the Spheron CLI / uses the API, **verifies with a throwaway
   create+destroy of a cheap GPU (e.g. a 4090)**, then deploys the A100, bootstraps it, and runs the experiment.
   The key is kept in an env var, never logged or committed.

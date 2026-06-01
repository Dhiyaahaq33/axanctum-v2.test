from __future__ import annotations

import time
from typing import Optional

from ..logging_setup import log
from ..models import RegimeContext


class RegimeEngine:
    """
    Market regime detector dengan persistence strength, confidence decay,
    dan asymmetric transition profiles.
    """

    # ── Transition profiles: responsiveness dan min_strength per pasangan ───────
    # responsiveness: seberapa cepat persistence_strength terakumulasi
    # min_strength:   batas minimum sebelum transisi diizinkan
    TRANSITION_PROFILES = {
        ("TRENDING",  "CHOP"):      {"responsiveness": 0.30, "min_strength": 0.60},
        ("TRENDING",  "EUPHORIC"):  {"responsiveness": 0.55, "min_strength": 0.50},
        ("TRENDING",  "PANIC"):     {"responsiveness": 0.85, "min_strength": 0.35},
        ("CHOP",      "TRENDING"):  {"responsiveness": 0.30, "min_strength": 0.65},
        ("CHOP",      "EUPHORIC"):  {"responsiveness": 0.45, "min_strength": 0.55},
        ("CHOP",      "PANIC"):     {"responsiveness": 0.90, "min_strength": 0.30},
        ("EUPHORIC",  "PANIC"):     {"responsiveness": 0.95, "min_strength": 0.25},
        ("EUPHORIC",  "TRENDING"):  {"responsiveness": 0.25, "min_strength": 0.70},
        ("EUPHORIC",  "CHOP"):      {"responsiveness": 0.35, "min_strength": 0.60},
        ("PANIC",     "RECOVERY"):  {"responsiveness": 0.40, "min_strength": 0.65},
        ("PANIC",     "CHOP"):      {"responsiveness": 0.30, "min_strength": 0.70},
        ("RECOVERY",  "TRENDING"):  {"responsiveness": 0.35, "min_strength": 0.65},
        ("RECOVERY",  "CHOP"):      {"responsiveness": 0.25, "min_strength": 0.72},
        ("RECOVERY",  "PANIC"):     {"responsiveness": 0.80, "min_strength": 0.40},
    }

    # ── Inertia base per regime (resistance awal untuk meninggalkan regime) ─────
    BASE_INERTIA = {
        "TRENDING":  0.55,
        "CHOP":      0.45,
        "EUPHORIC":  0.40,
        "PANIC":     0.20,   # keluar PANIC harus bisa cepat
        "RECOVERY":  0.50,
    }

    # RECOVERY hanya valid sebagai post-panic phase, bukan bullish default.
    RECOVERY_MEMORY_SEC = 12 * 3600

    def __init__(self, min_score_default: float = 70.0):
        self.min_score = min_score_default

        # ── State ───────────────────────────────────────────────────────────────
        self.current_regime:    str   = "TRENDING"   # regime aktif
        self.regime_entered_at: float = time.time()  # kapan regime dimulai

        self.candidate_regime:   Optional[str]   = None
        self.candidate_strength: float           = 0.0   # persistence strength

        self.last_scores:        dict = {}
        self.score_history:      list = []  # [{regime: score}, ...] rolling 6 scan
        self.last_panic_at:      Optional[float] = None

    # ─────────────────────────────────────────────────────────────────────────────
    # 1. SCORE REGIMES — dari data BTC-level
    # ─────────────────────────────────────────────────────────────────────────────

    def score_regimes(
        self,
        btc_price_vs_vwap:  float,   # % harga BTC vs VWAP-nya
        btc_cvd_spot:       float,   # CVD spot BTC (% delta)
        btc_cvd_fut:        float,   # CVD futures BTC
        btc_oi_delta:       float,   # OI delta BTC (%)
        btc_fr:             float,   # Funding rate BTC
        btc_fr_velocity:    float,   # Perubahan FR BTC vs 4 jam lalu
        btc_vol_ratio:      float,   # Volume ratio vs rata-rata
        liquidation_spike:  float,   # Volume liquidation (0 jika tidak ada data)
    ) -> dict:
        """
        Hitung raw score 0–100 untuk tiap regime.
        Score mencerminkan seberapa kuat bukti bahwa regime tersebut sedang aktif.
        """
        scores = {
            "TRENDING":  0.0,
            "CHOP":      0.0,
            "EUPHORIC":  0.0,
            "PANIC":     0.0,
            "RECOVERY":  0.0,
        }

        # ── TRENDING ────────────────────────────────────────────────────────────
        # Harga di atas VWAP + CVD spot positif + OI naik sehat + FR normal
        tr = 0.0
        if btc_price_vs_vwap > 0.5:
            tr += 20.0 + min(btc_price_vs_vwap / 5.0, 1.0) * 10.0
        if btc_cvd_spot > 0:
            tr += 15.0 + min(btc_cvd_spot / 8.0, 1.0) * 10.0
        if 0.5 <= btc_oi_delta <= 8.0:
            tr += 15.0
        if abs(btc_fr) < 0.0008:
            tr += 10.0  # FR normal = healthy
        if btc_vol_ratio >= 0.8:
            tr += 10.0  # volume stabil
        if btc_cvd_fut > 0 and btc_cvd_spot > 0:
            tr += 10.0  # keduanya positif = konfirmasi
        scores["TRENDING"] = min(tr, 100.0)

        # ── CHOP ────────────────────────────────────────────────────────────────
        # Price bolak-balik VWAP + volume rendah + delta tidak konsisten
        ch = 0.0
        if abs(btc_price_vs_vwap) < 1.5:
            ch += 25.0  # harga dekat VWAP = sideways
        if btc_vol_ratio < 0.7:
            ch += 20.0  # volume rendah
        if abs(btc_cvd_spot) < 2.0 and abs(btc_cvd_fut) < 2.0:
            ch += 20.0  # delta tidak konsisten
        if abs(btc_oi_delta) < 1.5:
            ch += 15.0  # OI random / tidak bergerak
        if abs(btc_fr) < 0.0003:
            ch += 10.0  # FR flat = tidak ada conviction
        # Konflik sinyal: CVD spot dan fut berlawanan arah = chop
        if btc_cvd_spot * btc_cvd_fut < 0:
            ch += 15.0
        scores["CHOP"] = min(ch, 100.0)

        # ── EUPHORIC ────────────────────────────────────────────────────────────
        # OI naik agresif + FR tinggi + harga jauh dari VWAP + CVD spot melemah
        eu = 0.0
        if btc_oi_delta > 5.0:
            eu += 20.0 + min((btc_oi_delta - 5.0) / 5.0, 1.0) * 15.0
        if btc_fr > 0.0010:
            eu += 20.0 + min((btc_fr - 0.001) / 0.001, 1.0) * 10.0
        if btc_price_vs_vwap > 3.0:
            eu += 15.0 + min((btc_price_vs_vwap - 3.0) / 5.0, 1.0) * 10.0
        if btc_fr_velocity > 0.0002:
            eu += 15.0  # FR naik cepat = FOMO masuk
        if btc_cvd_spot < btc_cvd_fut * 0.5:
            eu += 10.0  # spot tidak konfirmasi kenaikan futures
        scores["EUPHORIC"] = min(eu, 100.0)

        # ── PANIC ───────────────────────────────────────────────────────────────
        # OI collapse + liquidation spike + dump cepat + volume ekstrem
        pa = 0.0
        if btc_oi_delta < -5.0:
            pa += 30.0 + min(abs(btc_oi_delta + 5.0) / 5.0, 1.0) * 20.0
        if liquidation_spike > 0:
            pa += min(liquidation_spike / 10.0, 1.0) * 25.0
        if btc_price_vs_vwap < -3.0:
            pa += 20.0
        if btc_vol_ratio > 2.5:
            pa += 15.0  # volume ekstrem
        if btc_fr < -0.0005:
            pa += 10.0  # FR collapse / negatif
        scores["PANIC"] = min(pa, 100.0)

        # ── RECOVERY ────────────────────────────────────────────────────────────
        # Post-panic: OI mulai naik lagi + FR reset netral + CVD spot positif
        # Hanya masuk akal setelah PANIC — dikontrol oleh transition profile
        re = 0.0
        if 0.5 <= btc_oi_delta <= 4.0:
            re += 25.0  # OI naik tapi tidak agresif
        if -0.0003 <= btc_fr <= 0.0005:
            re += 25.0  # FR sudah reset netral
        if btc_cvd_spot > 1.0:
            re += 20.0  # spot mulai beli
        if btc_price_vs_vwap > -2.0 and btc_price_vs_vwap < 3.0:
            re += 15.0  # harga stabil, tidak ekstrem
        if btc_fr_velocity > -0.0001:
            re += 10.0  # FR tidak lagi turun

        now = time.time()
        recent_panic = (
            self.current_regime == "PANIC"
            or (
                self.last_panic_at is not None
                and (now - self.last_panic_at) <= self.RECOVERY_MEMORY_SEC
            )
        )

        # Recovery dikuatkan hanya kalau benar-benar post-panic.
        if self.current_regime == "PANIC":
            re *= 1.2
        elif not recent_panic:
            re = min(re, 35.0)
        scores["RECOVERY"] = min(re, 100.0)

        self.last_scores = scores
        self.score_history.append(dict(scores))
        if len(self.score_history) > 6:
            self.score_history.pop(0)

        return scores

    # ─────────────────────────────────────────────────────────────────────────────
    # 2. PERSISTENCE STRENGTH — bukan candidate count
    # ─────────────────────────────────────────────────────────────────────────────

    def calc_persistence_strength(
        self,
        candidate: str,
        current_scores: dict,
    ) -> float:
        """
        Akumulasi persistence strength berdasarkan:
        - dominance:   seberapa jauh candidate unggul dari regime lain
        - consistency: seberapa stabil sinyal pendukung di history
        - continuity:  apakah tidak ada interupsi (score tidak pernah < 30)

        3 weak confirmations ≠ 1 overwhelming signal.
        """
        if not self.score_history:
            return 0.0

        candidate_score = current_scores.get(candidate, 0.0)
        other_scores    = [v for k, v in current_scores.items() if k != candidate]
        second_best     = max(other_scores) if other_scores else 0.0

        # Dominance: selisih dengan regime terkuat lainnya
        dominance = max(0.0, (candidate_score - second_best) / 100.0)

        # Consistency: rata-rata score candidate di history
        hist_scores = [h.get(candidate, 0.0) for h in self.score_history]
        consistency = (sum(hist_scores) / len(hist_scores)) / 100.0

        # Continuity: tidak ada scan dimana score < 25 (tidak ada interupsi)
        continuity = 1.0 if all(h.get(candidate, 0.0) >= 25 for h in self.score_history) else 0.6

        # Strength = dominance × consistency × continuity
        strength = dominance * consistency * continuity

        # Boost untuk sinyal sangat overwhelming (PANIC dengan score > 80)
        if candidate == "PANIC" and candidate_score > 80:
            strength = max(strength, 0.85)

        return round(min(strength, 1.0), 4)

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. CONFIDENCE — dari ambiguity, bukan waktu
    # ─────────────────────────────────────────────────────────────────────────────

    def calc_confidence(
        self,
        current_scores: dict,
        btc_cvd_spot:  float,
        btc_cvd_fut:   float,
        btc_oi_delta:  float,
        btc_fr:        float,
    ) -> float:
        """
        Confidence berkurang karena ambiguity dari data, bukan karena waktu.
        Sumber decay:
        - CVD inconsistency: spot vs futures diverge
        - OI quality: bergerak tidak konsisten
        - FR elevation: mulai bergerak dari normal
        - Cross-signal conflict: regime-regime bersaing ketat
        """
        regime_score   = current_scores.get(self.current_regime, 50.0)
        base_confidence = regime_score / 100.0

        # ── Ambiguity sources ───────────────────────────────────────────────────
        # CVD inconsistency: spot dan fut berlawanan arah
        cvd_conflict = max(0.0, -(btc_cvd_spot * btc_cvd_fut)) / 20.0
        cvd_conflict = min(cvd_conflict, 0.20)

        # OI quality: bergerak terlalu random (sangat besar atau sangat kecil)
        oi_ambiguity = 0.0
        if abs(btc_oi_delta) > 12.0 or (0.0 < abs(btc_oi_delta) < 0.5):
            oi_ambiguity = 0.10

        # FR elevation dari zona normal
        fr_ambiguity = 0.0
        if abs(btc_fr) > 0.0012:
            fr_ambiguity = min((abs(btc_fr) - 0.0012) / 0.001, 0.15)

        # Cross-signal conflict: ada regime lain yang skornya berdekatan (< 15 poin)
        sorted_scores = sorted(current_scores.values(), reverse=True)
        regime_ambiguity = 0.0
        if len(sorted_scores) >= 2 and (sorted_scores[0] - sorted_scores[1]) < 15:
            regime_ambiguity = 0.15  # dua regime bersaing ketat

        total_decay = cvd_conflict + oi_ambiguity + fr_ambiguity + regime_ambiguity
        confidence  = base_confidence * (1.0 - min(total_decay, 0.45))

        return round(max(0.20, min(confidence, 1.0)), 3)

    # ─────────────────────────────────────────────────────────────────────────────
    # 4. EVALUATE TRANSITION — asimetris, inertia dari maturity
    # ─────────────────────────────────────────────────────────────────────────────

    def evaluate_transition(self, candidate: str, persistence_strength: float) -> bool:
        """
        Apakah transisi ke candidate diizinkan?
        - Inertia naik seiring maturity regime (tapi bisa di-override sinyal kuat)
        - Setiap pasangan punya responsiveness dan min_strength sendiri
        - PANIC: hampir selalu bisa override inertia
        """
        if candidate == self.current_regime:
            return False

        profile = self.TRANSITION_PROFILES.get(
            (self.current_regime, candidate),
            {"responsiveness": 0.40, "min_strength": 0.60}
        )

        # Minimum strength harus terpenuhi
        if persistence_strength < profile["min_strength"]:
            return False

        # Inertia scaling dari maturity regime
        maturity_hours = (time.time() - self.regime_entered_at) / 3600.0
        # Makin lama di regime, makin besar resistance (tapi tidak lebih dari 0.85)
        maturity_inertia = min(
            self.BASE_INERTIA[self.current_regime] + maturity_hours * 0.03,
            0.85
        )

        # Effective resistance setelah dimodifikasi oleh responsiveness candidate
        effective_resistance = maturity_inertia * (1.0 - profile["responsiveness"])

        # Transisi diizinkan jika strength > resistance
        return persistence_strength > effective_resistance

    # ─────────────────────────────────────────────────────────────────────────────
    # 5. UPDATE — satu langkah penuh per siklus scan
    # ─────────────────────────────────────────────────────────────────────────────

    def update(
        self,
        btc_price_vs_vwap: float,
        btc_cvd_spot:      float,
        btc_cvd_fut:       float,
        btc_oi_delta:      float,
        btc_fr:            float,
        btc_fr_velocity:   float,
        btc_vol_ratio:     float,
        liquidation_spike: float = 0.0,
    ) -> "RegimeContext":
        """
        Dipanggil sekali per siklus scan sebelum scan_coin dijalankan.
        Return: RegimeContext yang siap dipakai oleh scan pipeline.
        """
        # Score semua regime
        scores = self.score_regimes(
            btc_price_vs_vwap, btc_cvd_spot, btc_cvd_fut,
            btc_oi_delta, btc_fr, btc_fr_velocity,
            btc_vol_ratio, liquidation_spike,
        )

        # Tentukan candidate (regime dengan score tertinggi, bukan current)
        sorted_regimes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidate = sorted_regimes[0][0]
        if candidate == self.current_regime and len(sorted_regimes) > 1:
            candidate = sorted_regimes[1][0]

        # Hitung persistence strength candidate
        strength = self.calc_persistence_strength(candidate, scores)

        # Akumulasi atau reset candidate
        if candidate == self.candidate_regime:
            # Akumulasi dengan responsiveness dari transition profile
            profile = self.TRANSITION_PROFILES.get(
                (self.current_regime, candidate),
                {"responsiveness": 0.40, "min_strength": 0.60}
            )
            self.candidate_strength = min(
                self.candidate_strength + strength * profile["responsiveness"],
                1.0
            )
        else:
            # Candidate baru — reset
            self.candidate_regime  = candidate
            self.candidate_strength = strength

        # Evaluasi transisi
        if self.evaluate_transition(candidate, self.candidate_strength):
            log.info(
                f"[Regime] {self.current_regime} → {candidate} "
                f"(strength={self.candidate_strength:.2f})"
            )
            self.current_regime    = candidate
            self.regime_entered_at = time.time()
            if candidate == "PANIC":
                self.last_panic_at = self.regime_entered_at
            self.candidate_regime  = None
            self.candidate_strength = 0.0

        # Hitung confidence
        confidence = self.calc_confidence(
            scores, btc_cvd_spot, btc_cvd_fut, btc_oi_delta, btc_fr
        )

        log.info(
            f"[Regime] {self.current_regime} "
            f"confidence={confidence:.2f} "
            f"candidate={self.candidate_regime}({self.candidate_strength:.2f}) "
            f"scores={{{', '.join(f'{k[:3]}:{v:.0f}' for k,v in scores.items())}}}"
        )

        return self.get_context(confidence)

    # ─────────────────────────────────────────────────────────────────────────────
    # 6. GET_CONTEXT — full RegimeContext
    # ─────────────────────────────────────────────────────────────────────────────

    def get_context(self, confidence: float) -> "RegimeContext":
        """
        Return full RegimeContext berdasarkan regime aktif dan confidence.
        Semua nilai merupakan modifier terhadap default model.
        """
        base = self.min_score
        r    = self.current_regime

        # Confidence modifier: kalau confidence rendah, semua threshold naik
        conf_penalty = max(0.0, (0.65 - confidence) * 20.0)

        if r == "TRENDING":
            return RegimeContext(
                regime=r, confidence=confidence,
                long_threshold  = base - 3.0 + conf_penalty,   # sedikit lebih mudah
                short_threshold = base + 8.0 + conf_penalty,   # short lebih ketat
                w_cvd_spot=1.20, w_cvd_fut=1.10, w_oi=1.10, w_fr=0.90, w_vwap=0.95,
                aggressiveness          = 0.85,
                confirmation_bias       = 0.90,
                risk_profile            = "normal",
                allowed_setups          = ["continuation", "breakout", "pullback", "exhaustion_short"],
                exhaustion_sensitivity  = 0.70,   # short engine lebih tumpul
                continuation_trust      = 1.25,   # long lebih dipercaya
                score_bonus_continuation = 3.0,
                score_penalty_counter   = 8.0,
            )

        elif r == "CHOP":
            return RegimeContext(
                regime=r, confidence=confidence,
                long_threshold  = base + 8.0 + conf_penalty,
                short_threshold = base + 8.0 + conf_penalty,
                w_cvd_spot=0.75, w_cvd_fut=0.75, w_oi=0.70, w_fr=0.80, w_vwap=1.10,
                aggressiveness          = 0.45,
                confirmation_bias       = 1.40,
                risk_profile            = "reduced",
                allowed_setups          = ["extreme_reversal", "momentum_ignition"],
                exhaustion_sensitivity  = 0.90,
                continuation_trust      = 0.55,   # continuation tidak dipercaya
                score_bonus_continuation = 0.0,
                score_penalty_counter   = 5.0,
            )

        elif r == "EUPHORIC":
            return RegimeContext(
                regime=r, confidence=confidence,
                long_threshold  = base + 6.0 + conf_penalty,   # long lebih susah
                short_threshold = base - 5.0 + conf_penalty,   # short lebih mudah
                w_cvd_spot=0.90, w_cvd_fut=0.85, w_oi=1.30, w_fr=1.40, w_vwap=1.30,
                aggressiveness          = 0.70,
                confirmation_bias       = 1.10,
                risk_profile            = "normal",
                allowed_setups          = ["exhaustion_short", "fade_top"],
                exhaustion_sensitivity  = 1.50,   # short engine sangat sensitif
                continuation_trust      = 0.70,
                score_bonus_continuation = 0.0,
                score_penalty_counter   = 3.0,
            )

        elif r == "PANIC":
            return RegimeContext(
                regime=r, confidence=confidence,
                long_threshold  = 999.0,   # suspended — tidak ada long
                short_threshold = 999.0,   # suspended — tidak ada short baru
                w_cvd_spot=1.0, w_cvd_fut=1.0, w_oi=1.0, w_fr=1.0, w_vwap=1.0,
                aggressiveness          = 0.0,
                confirmation_bias       = 2.0,
                risk_profile            = "suspended",
                allowed_setups          = [],   # tidak ada setup yang diizinkan
                exhaustion_sensitivity  = 1.0,
                continuation_trust      = 0.0,
                score_bonus_continuation = 0.0,
                score_penalty_counter   = 20.0,
            )

        else:  # RECOVERY
            return RegimeContext(
                regime=r, confidence=confidence,
                long_threshold  = base - 5.0 + conf_penalty,   # long paling mudah
                short_threshold = base + 12.0 + conf_penalty,  # short sangat ketat
                w_cvd_spot=1.30, w_cvd_fut=1.10, w_oi=1.15, w_fr=1.20, w_vwap=1.00,
                aggressiveness          = 0.80,
                confirmation_bias       = 1.00,
                risk_profile            = "normal",
                allowed_setups          = ["continuation", "accumulation_long"],
                exhaustion_sensitivity  = 0.50,   # short engine hampir mati
                continuation_trust      = 1.40,   # long paling dipercaya
                score_bonus_continuation = 5.0,
                score_penalty_counter   = 12.0,
            )


# ── Global instance — satu engine untuk seluruh proses ──────────────────────
_regime_engine: Optional[RegimeEngine] = None

def get_regime_engine(min_score: float = 70.0) -> RegimeEngine:
    global _regime_engine
    if _regime_engine is None:
        _regime_engine = RegimeEngine(min_score_default=min_score)
    return _regime_engine

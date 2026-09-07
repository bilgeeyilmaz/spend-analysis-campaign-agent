"""Denetim izi (audit log) — her agent cevabı JSONL olarak diske yazılır.

    from src.audit.logger import kaydet
    kaydet(cevap, config)

NEDEN VAR: bankada "model bu müşteriye neden bu kampanyayı önerdi" sorusunun
cevabı **sonradan** verilebilmelidir. Bir öneri hakkında şikâyet geldiğinde
hangi verinin okunduğu, hangi modelin çalıştığı, hangi araçların çağrıldığı ve
denetimin ne dediği kayıttan okunabilmeli. Açıklanabilirlik anlatısının
somut karşılığı bu dosyadır.

NE YAZILIR: modelin GÖRDÜĞÜ veri (yani `mask_payload`'dan geçmiş hâli), araç
çağrıları, denetim ihlalleri, hangi sağlayıcının cevap verdiği ve yedeğe düşülüp
düşülmediği. Ham işlem listesi yazılmaz — kayıt bir kopya veri ambarı değil,
karar izidir.

NE YAZILMAZ: kişisel veri. Şemada zaten yok (`schemas.Customer`), payload da
`mask_payload`'dan geçiyor; bu dosya üçüncü halka. `customer_id` yazılır çünkü
denetim izinin anlamı odur — o bir kimlik değil, anahtardır.

JSONL seçildi çünkü satır bazlı eklemeli yazma en ucuz ve en dayanıklı biçim:
süreç ortada ölürse önceki satırlar bozulmaz. Analiz için `pandas.read_json(...,
lines=True)` tek satırda okur.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import resolve_path

logger = logging.getLogger(__name__)


def _kayit_olustur(cevap, config: dict) -> dict:
    """AgentCevabi -> JSONL satırı (sözlük)."""
    agent_cfg = config.get("agent", {})
    oneriler = [
        oneri
        for cagri in cevap.tool_calls
        if cagri.get("name") == "score_campaigns"
        for oneri in (cagri.get("result") or {}).get("oneriler", [])
    ]

    return {
        "zaman": datetime.now(timezone.utc).isoformat(),
        "customer_id": cevap.customer_id,
        "provider": cevap.provider,
        "model": agent_cfg.get("groq", {}).get("model") if cevap.provider == "groq" else None,
        "iterations": cevap.iterations,
        # Yedeğe düşüldüyse cevabı LLM değil şablon üretmiştir; sonradan
        # "o gün model ne dedi" diye bakıldığında bu ayrım şart.
        "fallback": cevap.fallback,
        "uyarilar": cevap.uyarilar,
        "arac_cagrilari": [
            {"name": cagri.get("name"), "arguments": cagri.get("arguments")}
            for cagri in cevap.tool_calls
        ],
        "onerilen_kampanyalar": [
            {"campaign_id": o.get("campaign_id"), "score": o.get("score")}
            for o in oneriler
        ],
        "answer": cevap.answer,
    }


def kaydet(cevap, config: dict) -> Path | None:
    """Cevabı audit log'a ekler. Kapalıysa ya da yazamazsa sessizce geçer.

    Log yazamamak bir cevabı engellemez: denetim izi tutmak önemlidir ama
    müşteriye cevap verememekten daha önemli değildir. Hata loglanır.
    """
    audit_cfg = config.get("audit", {})
    if not audit_cfg.get("enabled", False):
        return None

    path = resolve_path(audit_cfg.get("log_path", "logs/audit.jsonl"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        satir = json.dumps(_kayit_olustur(cevap, config), ensure_ascii=False, default=str)
        with path.open("a", encoding="utf-8") as f:
            f.write(satir + "\n")
        return path
    except OSError as hata:
        logger.warning("Audit log yazılamadı (%s): %s", path, hata)
        return None


def oku(config: dict):
    """Audit log'u DataFrame olarak okur (değerlendirme ve demo için)."""
    import pandas as pd

    path = resolve_path(config.get("audit", {}).get("log_path", "logs/audit.jsonl"))
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)

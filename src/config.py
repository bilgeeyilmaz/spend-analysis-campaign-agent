"""Konfigürasyon yükleyici.

Kod içinde sabit yol/parametre bulunmaması için her şey config.yaml'dan okunur.
Gizli bilgiler (API anahtarı) config'e DEĞİL, .env dosyasına yazılır.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

#: Proje kök dizini — src/config.py'nin iki üstü.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


#: Konfigürasyon yolunu ezen ortam değişkeni. Komut satırı araçlarının hepsi
#: `load_config()`'i argümansız çağırır; bu değişken olmadan tüm zinciri
#: (üreteç -> özellik -> model -> API) gerçek `data/` dizinine dokunmadan
#: çalıştırmanın yolu yoktu. Uçtan uca test ve demo bunu kullanır.
CONFIG_ENV = "KAMPANYA_CONFIG"


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """config.yaml'ı okur ve sözlük olarak döner (bir kez okunup önbelleklenir).

    Yol önceliği: açık argüman > `KAMPANYA_CONFIG` > proje kökündeki config.yaml.
    """
    ortam = os.getenv(CONFIG_ENV, "").strip()
    cfg_path = Path(path) if path else Path(ortam) if ortam else PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Konfigürasyon bulunamadı: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str | Path) -> Path:
    """config.yaml'daki göreli yolu proje köküne göre mutlak yola çevirir.

    Böylece komutlar hangi dizinden çalıştırılırsa çalıştırılsın aynı sonucu verir.
    """
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def get_groq_api_key() -> str | None:
    """Groq anahtarını .env'den okur. Yoksa None döner ve sistem yedek moda geçer."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    return key or None

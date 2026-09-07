"""Audit log testleri.

Denetim izinin varlık sebebi: "model bu müşteriye neden bu kampanyayı önerdi"
sorusunun SONRADAN cevaplanabilmesi. Testler bunun için gereken üç şeyi
doğruluyor — kayıt yazılıyor mu, içinde karar izi var mı, ve log yazılamadığında
cevap üretimi engelleniyor mu (engellenmemeli).
"""

from __future__ import annotations

import copy
import json

import pytest

from src.agent.orchestrator import AgentCevabi
from src.audit.logger import _kayit_olustur, kaydet, oku
from src.config import load_config

CEVAP = AgentCevabi(
    customer_id="C000004",
    answer="Son 3 ayda 53.609 TL harcadınız.",
    provider="groq",
    tool_calls=[
        {"name": "analyze_spending", "arguments": {"customer_id": "C000004"},
         "result": {"toplam_harcama": 53609.49}},
        {"name": "score_campaigns", "arguments": {"customer_id": "C000004"},
         "result": {"oneriler": [{"campaign_id": "KMP001", "score": 0.53}]}},
    ],
    iterations=3,
    fallback=False,
    uyarilar=[],
)


@pytest.fixture
def config(tmp_path) -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["audit"] = {"enabled": True, "log_path": str(tmp_path / "audit.jsonl")}
    return cfg


def test_kayit_karar_izini_tasir(config):
    kayit = _kayit_olustur(CEVAP, config)
    assert kayit["customer_id"] == "C000004"
    assert kayit["provider"] == "groq"
    assert kayit["model"] == config["agent"]["groq"]["model"]
    assert kayit["onerilen_kampanyalar"] == [{"campaign_id": "KMP001", "score": 0.53}]
    assert [c["name"] for c in kayit["arac_cagrilari"]] == [
        "analyze_spending", "score_campaigns"]


def test_yedege_dusulduyse_kayitta_gorunur(config):
    """"O gün model ne dedi" sorusunun cevabı buna bağlı."""
    yedek = copy.deepcopy(CEVAP)
    yedek.fallback, yedek.provider = True, "mock"
    kayit = _kayit_olustur(yedek, config)
    assert kayit["fallback"] is True
    assert kayit["model"] is None, "mock cevabında model adı yazılmamalı"


def test_jsonl_satir_satir_eklenir(config):
    kaydet(CEVAP, config)
    kaydet(CEVAP, config)
    path = kaydet(CEVAP, config)
    satirlar = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(satirlar) == 3
    assert json.loads(satirlar[0])["customer_id"] == "C000004"


def test_kapaliysa_yazilmaz(config):
    config["audit"]["enabled"] = False
    assert kaydet(CEVAP, config) is None


def test_yazilamazsa_cevap_engellenmez(config):
    """Log tutmak önemli ama müşteriye cevap verememekten daha önemli değil."""
    config["audit"]["log_path"] = "/olmayan-dizin/audit.jsonl"
    assert kaydet(CEVAP, config) is None      # istisna fırlatmaz


def test_oku_dataframe_dondurur(config):
    kaydet(CEVAP, config)
    df = oku(config)
    assert len(df) == 1
    assert df.iloc[0]["customer_id"] == "C000004"


def test_agent_cevabi_otomatik_kaydediliyor(config, tmp_path):
    """Dört ayrı çıkış yolu var; hepsi `_bitir`'den geçmeli."""
    from src.agent.llm import MockProvider
    from src.agent.orchestrator import Agent

    agent = Agent(config, provider=MockProvider())
    cid = agent.ctx.source.get_customers()["customer_id"].iloc[0]
    agent.recommend(cid)
    assert oku(config).shape[0] == 1

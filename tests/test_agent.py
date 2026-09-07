"""Agent orkestratörü, sağlayıcılar ve şablon testleri.

Testlerin tamamı **ağsızdır** — `MockProvider` bir test taklidi değil, sistemin
üretim yedeği olduğu için gerçek yol zaten test edilmiş oluyor.

Buradaki asıl soru "LLM güzel cümle kurdu mu" değil (onu test edemeyiz), şu üç
şey: (1) döngü araçları doğru sırayla çağırıyor mu, (2) hata durumlarında
çöküyor mu, (3) uygunluk kararı gerçekten LLM'in elinde değil mi.
"""

from __future__ import annotations

import copy
import shutil

import pytest

from src.agent.llm import LLMResponse, MockProvider, ToolCall, build_provider
from src.agent.orchestrator import SISTEM_PROMPTU, Agent
from src.agent import templates
from src.agent.tools import ToolContext
from src.config import PROJECT_ROOT, load_config
from src.data import generator
from src.features.build import build_feature_table
from src.models.propensity import train

N_CUSTOMERS = 220


@pytest.fixture(scope="module")
def agent(tmp_path_factory) -> Agent:
    raw_dir = tmp_path_factory.mktemp("raw")
    processed_dir = tmp_path_factory.mktemp("processed")
    model_dir = tmp_path_factory.mktemp("models")
    shutil.copy(PROJECT_ROOT / "data" / "raw" / "campaigns.json", raw_dir / "campaigns.json")
    assert generator.main([
        "--n-customers", str(N_CUSTOMERS), "--out", str(raw_dir),
        "--as-of", "2026-08-13", "--seed", "7",
    ]) == 0

    config = copy.deepcopy(load_config())
    config["data"]["json"]["raw_dir"] = str(raw_dir)
    config["data"]["processed_dir"] = str(processed_dir)
    config["models"]["dir"] = str(model_dir)
    config["agent"]["provider"] = "mock"
    config["models"]["propensity"]["param_grid"] = {
        "learning_rate": [0.05], "num_leaves": [4], "n_estimators": [100],
    }
    build_feature_table(config).to_parquet(processed_dir / "customer_features.parquet", index=False)

    ctx = ToolContext(config)
    paket, _ = train(config, algorithm="hist")
    ctx._propensity = paket
    return Agent(config, ctx=ctx)


def _musteri(agent: Agent, **kosul) -> str:
    df = agent.ctx.source.get_customers()
    for alan, deger in kosul.items():
        df = df[df[alan] == deger]
    assert not df.empty, f"koşulu sağlayan müşteri yok: {kosul}"
    return df.iloc[0]["customer_id"]


# --------------------------------------------------------------------------
# Sağlayıcı seçimi
# --------------------------------------------------------------------------


def test_anahtar_yoksa_yedege_dusulur(monkeypatch):
    """`provider: groq` yazsa bile anahtar yoksa sistem çalışmaya devam etmeli."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    config = copy.deepcopy(load_config())
    config["agent"]["provider"] = "groq"
    assert build_provider(config).name == "mock"


def test_mock_acikca_secilebilir():
    config = copy.deepcopy(load_config())
    config["agent"]["provider"] = "mock"
    assert build_provider(config).name == "mock"


# --------------------------------------------------------------------------
# Döngü: araç sırası ve çok adımlılık
# --------------------------------------------------------------------------


def test_oneri_akisi_araclari_sirayla_cagirir(agent):
    cevap = agent.recommend(_musteri(agent, opt_in_marketing=True))
    adlar = [c["name"] for c in cevap.tool_calls]
    assert adlar[0] == "analyze_spending", "analiz her zaman önce gelir"
    assert "score_campaigns" in adlar
    assert cevap.iterations > 1, "tek turda biterse tool-calling döngüsü çalışmıyor demektir"


def test_serbest_soru_kampanya_araclarini_cagirmaz(agent):
    """Harcama sorusunda kampanya skorlamak gereksiz iş — agent karar veriyor."""
    cevap = agent.chat(_musteri(agent), "geçen ay en çok neye harcadım?")
    adlar = {c["name"] for c in cevap.tool_calls}
    assert adlar == {"analyze_spending"}


def test_kampanya_sorusu_skorlamayi_tetikler(agent):
    cevap = agent.chat(_musteri(agent, opt_in_marketing=True), "bana uygun bir kampanya var mı?")
    assert "score_campaigns" in {c["name"] for c in cevap.tool_calls}


# --------------------------------------------------------------------------
# Dayanıklılık: demo hiçbir koşulda çökmemeli
# --------------------------------------------------------------------------


class PatlayanProvider:
    name = "patlayan"

    def chat(self, messages, tools=None):
        raise RuntimeError("429 rate limit")


def test_llm_patlarsa_yedege_dusulur(agent):
    """Groq limiti dolduğunda kullanıcı hata değil cevap görmeli."""
    kirik = Agent(agent.config, ctx=agent.ctx, provider=PatlayanProvider())
    cevap = kirik.recommend(_musteri(agent, opt_in_marketing=True))
    assert cevap.fallback is True
    assert cevap.answer.strip()
    assert any(u.startswith("llm_hatasi") for u in cevap.uyarilar)


class BosCevapProvider:
    name = "bos"

    def chat(self, messages, tools=None):
        return LLMResponse(content="   ", provider=self.name)


def test_bos_llm_cevabi_deterministik_metne_duser(agent):
    bos = Agent(agent.config, ctx=agent.ctx, provider=BosCevapProvider())
    cevap = bos.recommend(_musteri(agent, opt_in_marketing=True))
    assert cevap.fallback is True
    assert "bos_llm_cevabi" in cevap.uyarilar
    assert len(cevap.answer) > 20


class SonsuzDonguProvider:
    name = "sonsuz"

    def chat(self, messages, tools=None):
        return LLMResponse(
            tool_calls=[ToolCall("x", "analyze_spending", {"customer_id": "C000000"})],
            provider=self.name,
        )


def test_sonsuz_dongu_ust_sinirda_durur(agent):
    sonsuz = Agent(agent.config, ctx=agent.ctx, provider=SonsuzDonguProvider())
    cevap = sonsuz.recommend(_musteri(agent, opt_in_marketing=True))
    assert cevap.iterations == sonsuz.max_iter
    assert "max_iterations" in cevap.uyarilar
    assert cevap.answer.strip(), "sınıra dayanınca da bir cevap üretilmeli"


class UydurmaAracProvider:
    name = "uydurma"

    def __init__(self):
        self.tur = 0

    def chat(self, messages, tools=None):
        self.tur += 1
        if self.tur == 1:
            return LLMResponse(tool_calls=[ToolCall("x", "delete_customer", {})],
                               provider=self.name)
        return LLMResponse(content="Analiz tamamlandı.", provider=self.name)


def test_uydurma_arac_dongusu_cokertmez(agent):
    """Var olmayan araç, hata mesajı olarak LLM'e döner; döngü devam eder."""
    uydurma = Agent(agent.config, ctx=agent.ctx, provider=UydurmaAracProvider())
    cevap = uydurma.recommend(_musteri(agent))
    assert any(u.startswith("arac_hatasi") for u in cevap.uyarilar)
    assert cevap.answer == "Analiz tamamlandı."


# --------------------------------------------------------------------------
# Uygunluk kararı LLM'de değil
# --------------------------------------------------------------------------


def test_izinsiz_musteriye_kampanya_onerilmez(agent):
    cevap = agent.recommend(_musteri(agent, opt_in_marketing=False))
    katalog = set(agent.ctx.source.get_campaigns()["campaign_id"])
    assert not any(kod in cevap.answer for kod in katalog)
    assert "uygun bir kampanya bulunmuyor" in cevap.answer


class UydurmaKampanyaProvider:
    name = "uydurma_kampanya"

    def chat(self, messages, tools=None):
        return LLMResponse(content="Size KMP014 kampanyasını öneriyorum.", provider=self.name)


def test_izinsiz_kampanya_capraz_kontrolde_yakalanir(agent):
    """LLM elenmiş bir kampanyayı metne sokarsa uyarı üretilir."""
    kacak = Agent(agent.config, ctx=agent.ctx, provider=UydurmaKampanyaProvider())
    cevap = kacak.recommend(_musteri(agent, opt_in_marketing=True))
    assert "izinsiz_kampanya: KMP014" in cevap.uyarilar


def test_sistem_promptu_yasaklari_iceriyor():
    """Uyum kuralları promptta yazılı olmalı — Gün 8 bunları ayrıca denetleyecek."""
    for yasak in ("tavsiye", "uydurma", "check_eligibility"):
        assert yasak in SISTEM_PROMPTU


# --------------------------------------------------------------------------
# Şablonlar (LLM'siz metin)
# --------------------------------------------------------------------------


def test_ozet_analiz_verisini_kullanir():
    analiz = {
        "islem_var": True, "period_months": 3, "toplam_harcama": 12345.0,
        "islem_adedi": 42, "trend": {},
        "kategori_dagilimi": [{"kategori": "market", "pay": 0.4, "tutar": 4938.0}],
        "duzenli_giderler": [], "olagandisi_artislar": [],
    }
    metin = templates.harcama_ozeti(analiz)
    assert "12.345 TL" in metin and "42 işlem" in metin and "market %40" in metin


def test_ozet_islem_yoksa_anlamli_cumle_kurar():
    assert "bulunamadı" in templates.harcama_ozeti({"islem_var": False})


def test_gerekce_hedef_kategoriden_turer():
    analiz = {
        "olagandisi_artislar": [], "duzenli_giderler": [],
        "kategori_dagilimi": [{"kategori": "seyahat", "pay": 0.22}],
    }
    gerekce = templates.kampanya_gerekcesi(analiz, {"target_categories": ["seyahat"]})
    assert "seyahat" in gerekce and "%22" in gerekce


def test_gerekce_ilgisiz_kampanyada_uydurmaz():
    """Bağ kurulamıyorsa genel cümleye düşer; sahte gerekçe yazmaz."""
    analiz = {
        "olagandisi_artislar": [], "duzenli_giderler": [],
        "kategori_dagilimi": [{"kategori": "market", "pay": 0.9}],
    }
    gerekce = templates.kampanya_gerekcesi(analiz, {"target_categories": ["seyahat"]})
    assert "seyahat" not in gerekce


def test_tl_bicimi_turkce():
    assert templates.tl(1234567.8) == "1.234.568 TL"


def test_kampanya_yoksa_analiz_yine_de_anlatilir():
    analiz = {
        "islem_var": True, "period_months": 3, "toplam_harcama": 5000.0,
        "islem_adedi": 10, "trend": {}, "kategori_dagilimi": [],
        "duzenli_giderler": [], "olagandisi_artislar": [],
    }
    metin = templates.kampanya_metni(analiz, [], {})
    assert "5.000 TL" in metin
    assert "uygun bir kampanya bulunmuyor" in metin


def test_metin_tavsiye_icermez():
    """Agent betimler, tavsiye vermez — yatırım tavsiyesi düzenlemeye tabidir."""
    analiz = {
        "islem_var": True, "period_months": 3, "toplam_harcama": 50000.0,
        "islem_adedi": 40, "trend": {},
        "kategori_dagilimi": [{"kategori": "market", "pay": 0.5}],
        "duzenli_giderler": [
            {"kategori": "telekom", "ay_sayisi": 6, "ortalama_tutar": 450.0}
        ],
        "olagandisi_artislar": [
            {"tip": "buyuk_tek_alim", "kategori": "elektronik", "tutar": 30000.0,
             "kat": 12.0, "donem": "son 30 gün", "taksitli": True}
        ],
    }
    metin = templates.harcama_ozeti(analiz).lower()
    for yasak in ("iptal ed", "tasarruf", "yatırım", "azalt", "öneririz ki", "dikkat edin"):
        assert yasak not in metin, f"tavsiye ifadesi sızmış: {yasak}"


@pytest.mark.parametrize("ham, beklenen", [
    ("%%1,8 düşüş", "%1,8 düşüş"),
    ("iki  boşluk", "iki boşluk"),
    ("%25 nakit iade", "%25 nakit iade"),
])
def test_bicim_duzeltme_yalniz_kozmetik(ham, beklenen):
    """Gözlenen kusur: model "%%1,8" yazıyordu. Düzeltme yalnız biçimsel —
    sayıya dokunmak denetimin izlenebilirlik kontrolünü yanıltırdı."""
    from src.agent.orchestrator import _bicim_duzelt

    assert _bicim_duzelt(ham) == beklenen


def test_bicim_duzeltme_sayilari_korur():
    from src.agent.orchestrator import _bicim_duzelt

    metin = "Son 3 ayda 24.366,85 TL harcadınız; %27'si elektronik."
    assert _bicim_duzelt(metin) == metin

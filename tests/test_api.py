"""FastAPI servisi testleri.

API ince bir katman olduğu için burada iş kuralı test edilmez (o `test_tools`
ve `test_agent`'ın işi). Burada sorulan üç şey var:

  1. **Sözleşme** — HTTP gövdesi `AgentCevabi` ile aynı alanları taşıyor mu?
     Orkestratöre yeni bir alan eklenip API şeması güncellenmezse, alan
     istemciye sessizce ulaşmaz. Bunu yakalayan bir test yoksa fark edilmez.
  2. **Hata sınıflandırması** — olmayan müşteri 404 mü, servis hazır değilse
     503 mü? Hepsi 500 dönen bir servis, hata ayıklamayı log okumaya çevirir.
  3. **Sınırın kendisi** — PII yokluğu ve `opt_in_marketing` kapısı HTTP
     çıktısında da geçerli mi? Projenin hukuki iddiası, iç fonksiyonda değil,
     dışarı çıkan gövdede doğrulanmalı.

Testler ağsızdır ve `data/raw/` altındaki üretilmiş veriye dokunmaz: kendi
küçük veri setini üretip ajanı bağımlılık geçersizlemesiyle enjekte eder.
Uygulama ömrü (lifespan) bilerek çalıştırılmaz — 80 MB JSON'u her test
oturumunda okumanın karşılığı yok.
"""

from __future__ import annotations

import copy
import dataclasses
import shutil

import pytest
from fastapi.testclient import TestClient

from src.agent.orchestrator import Agent, AgentCevabi
from src.agent.tools import ToolContext
from src.api.main import AjanYaniti, app, get_agent
from src.config import PROJECT_ROOT, load_config
from src.data import generator
from src.data.validate import PII_COLUMNS
from src.features.build import build_feature_table
from src.models.propensity import train

N_CUSTOMERS = 220


@pytest.fixture(scope="module")
def agent(tmp_path_factory) -> Agent:
    """`tests/test_agent.py` ile aynı kurulum: küçük veri, mock sağlayıcı."""
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


@pytest.fixture
def client(agent) -> TestClient:
    """Hazır servis. `with` kullanılmıyor: lifespan çalışmasın, gerçek
    `data/raw` okunmasın — ajanı zaten biz veriyoruz."""
    app.dependency_overrides[get_agent] = lambda: agent
    app.state.agent = agent
    app.state.config = agent.config
    app.state.hata = None
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
        app.state.agent = None


@pytest.fixture
def bozuk_client() -> TestClient:
    """Ajanın kurulamadığı servis (eksik parquet, bozuk model...)."""
    app.dependency_overrides.clear()
    app.state.agent = None
    app.state.config = load_config()
    app.state.hata = "FileNotFoundError: customer_features.parquet yok"
    return TestClient(app)


def _musteri(agent: Agent, **kosul) -> str:
    df = agent.ctx.source.get_customers()
    for alan, deger in kosul.items():
        df = df[df[alan] == deger]
    assert not df.empty, f"koşulu sağlayan müşteri yok: {kosul}"
    return df.iloc[0]["customer_id"]


# --------------------------------------------------------------------------
# Sözleşme: gövde orkestratörle aynı alanları taşımalı
# --------------------------------------------------------------------------


def test_yanit_semasi_agent_cevabi_ile_ayni_alanlari_tasiyor():
    """API kendi cevap şeklini tanımlamaz, `AgentCevabi`'yi yansıtır.

    Orkestratöre alan eklenip burası güncellenmezse (ya da tersi) FastAPI
    fazlalığı sessizce kırpar. Kırılması gereken yer burasıdır.
    """
    dataclass_alanlari = {f.name for f in dataclasses.fields(AgentCevabi)}
    sema_alanlari = set(AjanYaniti.model_fields)
    assert sema_alanlari == dataclass_alanlari


def test_recommend_tam_govde_dondurur(client, agent):
    cid = _musteri(agent, opt_in_marketing=True)

    r = client.post("/recommend", json={"customer_id": cid})

    assert r.status_code == 200
    govde = r.json()
    assert set(govde) == set(AjanYaniti.model_fields)
    assert govde["customer_id"] == cid
    assert govde["answer"].strip()
    assert govde["tool_calls"], "gerekçe (araç çağrıları) boş dönmemeli"


def test_chat_cevap_uretir(client, agent):
    cid = _musteri(agent, opt_in_marketing=True)

    r = client.post("/chat", json={"customer_id": cid, "question": "Geçen ay nereye harcadım?"})

    assert r.status_code == 200
    assert r.json()["answer"].strip()


def test_profil_ucu_kunye_dondurur(client, agent):
    cid = _musteri(agent, opt_in_marketing=True)

    r = client.get(f"/customers/{cid}/profile")

    assert r.status_code == 200
    assert r.json()["customer_id"] == cid


# --------------------------------------------------------------------------
# Sınır: PII ve izin kapısı HTTP çıktısında da geçerli
# --------------------------------------------------------------------------


def test_profil_ucunda_pii_yok(client, agent):
    """Sızıntının olabileceği son nokta burası: içeride temiz olması yetmez."""
    cid = _musteri(agent, opt_in_marketing=True)

    govde = client.get(f"/customers/{cid}/profile").json()

    sizan = set(govde) & set(PII_COLUMNS)
    assert not sizan, f"künyede PII alanı var: {sizan}"


def test_izin_vermeyen_musteriye_kampanya_onerilmez(client, agent):
    """`opt_in_marketing=False` yasal kapıdır; HTTP cevabında kampanya kodu
    veya öneri geçmemeli."""
    cid = _musteri(agent, opt_in_marketing=False)

    govde = client.post("/recommend", json={"customer_id": cid}).json()

    assert govde["customer_id"] == cid
    katalog = set(agent.ctx.source.get_campaigns()["campaign_id"])
    assert not [k for k in katalog if k in govde["answer"]]
    assert not govde["uyarilar"], f"denetim uyarısı: {govde['uyarilar']}"


# --------------------------------------------------------------------------
# Hata sınıflandırması
# --------------------------------------------------------------------------


def test_olmayan_musteri_404(client):
    r = client.post("/recommend", json={"customer_id": "C999999"})

    assert r.status_code == 404
    assert "C999999" in r.json()["detail"]


def test_olmayan_musteri_profil_404(client):
    assert client.get("/customers/C999999/profile").status_code == 404


def test_bos_customer_id_422(client):
    assert client.post("/recommend", json={"customer_id": ""}).status_code == 422


def test_eksik_soru_422(client, agent):
    cid = _musteri(agent, opt_in_marketing=True)
    assert client.post("/chat", json={"customer_id": cid}).status_code == 422


def test_asiri_uzun_soru_422(client, agent):
    """Sınırsız metin hem token maliyeti hem prompt injection yüzeyidir."""
    cid = _musteri(agent, opt_in_marketing=True)
    r = client.post("/chat", json={"customer_id": cid, "question": "a" * 5_000})
    assert r.status_code == 422


def test_beklenmeyen_hata_500_ama_ayrinti_sizmaz(client, agent, monkeypatch):
    """Yığın izi istemciye gitmemeli; log'a gider."""
    def patla(_cid):
        raise RuntimeError("veritabanı parolası: hunter2")

    monkeypatch.setattr(agent, "recommend", patla)
    r = client.post("/recommend", json={"customer_id": "C000001"})

    assert r.status_code == 500
    assert "hunter2" not in r.text
    assert "RuntimeError" in r.json()["detail"]


def test_ajan_hazir_degilse_503(bozuk_client):
    """Servis ayakta ama bağımlılık eksik: 500 değil 503, üstelik ne yapılacağını
    söyleyen bir mesajla."""
    r = bozuk_client.post("/recommend", json={"customer_id": "C000001"})

    assert r.status_code == 503
    assert "src.features.build" in r.json()["detail"]


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------


def test_health_hazir_serviste_ok(client):
    r = client.get("/health")

    assert r.status_code == 200
    govde = r.json()
    assert govde["status"] == "ok"
    assert all(d["hazir"] for d in govde["bagimliliklar"].values())
    assert set(govde["bagimliliklar"]) == {"veri_kaynagi", "ozellik_tablosu", "propensity_modeli"}


def test_health_mock_saglayiciyi_ariza_saymaz(client):
    """LLM'siz çalışmak tasarlanmış davranış; "degraded" demek yanlış alarmdır."""
    govde = client.get("/health").json()

    assert govde["provider"] == "MockProvider"
    assert "yedek" in govde["llm"]
    assert govde["status"] == "ok"


def test_health_ajan_yokken_de_cevap_verir(bozuk_client):
    """Sağlık ucu, servis bozukken de cevap verebilmek içindir."""
    r = bozuk_client.get("/health")

    assert r.status_code == 200
    govde = r.json()
    assert govde["status"] == "degraded"
    assert "FileNotFoundError" in govde["bagimliliklar"]["agent"]["ayrinti"]


# --------------------------------------------------------------------------
# Harcama analizi ucu
# --------------------------------------------------------------------------


def test_harcama_ucu_analiz_dondurur(client, agent):
    cid = _musteri(agent, opt_in_marketing=True)

    r = client.get(f"/customers/{cid}/spending")

    assert r.status_code == 200
    govde = r.json()
    assert govde["customer_id"] == cid
    for alan in ("kategori_dagilimi", "aylik_seri", "trend", "duzenli_giderler"):
        assert alan in govde, f"analiz çıktısında {alan} yok"


def test_harcama_ucu_kampanya_katalogunu_okumaz(client, agent):
    """Analiz, önerinin yan ürünü değildir: çıktısında kampanya izi olmamalı."""
    cid = _musteri(agent, opt_in_marketing=True)
    katalog = set(agent.ctx.source.get_campaigns()["campaign_id"])

    metin = client.get(f"/customers/{cid}/spending").text

    assert not [k for k in katalog if k in metin]
    assert "campaign" not in metin.lower()


def test_harcama_ucu_gecersiz_pencereyi_reddeder(client, agent):
    cid = _musteri(agent, opt_in_marketing=True)
    assert client.get(f"/customers/{cid}/spending?months=0").status_code == 422
    assert client.get(f"/customers/{cid}/spending?months=3").status_code == 200


def test_harcama_ucu_olmayan_musteri_404(client):
    assert client.get("/customers/C999999/spending").status_code == 404


# --------------------------------------------------------------------------
# Müşteri listesi ucu
# --------------------------------------------------------------------------


def test_musteri_listesi_kimlik_donduruyor(client, agent):
    r = client.get("/customers?limit=5")

    assert r.status_code == 200
    govde = r.json()
    assert govde["dondurulen"] == 5
    assert govde["toplam"] == len(agent.ctx.source.get_customers())
    assert set(govde["musteriler"][0]) == {
        "customer_id", "customer_segment", "opt_in_marketing"
    }


def test_musteri_listesi_beyaz_liste_disina_cikmiyor(client):
    """Kaynak tabloda gender ve city de var; bu uçtan çıkmamalılar."""
    metin = client.get("/customers?limit=50").text

    assert "gender" not in metin
    assert "city" not in metin


def test_musteri_listesi_izne_gore_suzuluyor(client):
    """Arayüz "izin vermeyen müşteri" senaryosunu gösterebilmeli."""
    izinli = client.get("/customers?limit=2000&opted_in=true").json()
    izinsiz = client.get("/customers?limit=2000&opted_in=false").json()

    assert all(m["opt_in_marketing"] for m in izinli["musteriler"])
    assert not any(m["opt_in_marketing"] for m in izinsiz["musteriler"])
    assert izinli["toplam"] + izinsiz["toplam"] == client.get("/customers?limit=1").json()["toplam"]


def test_musteri_listesi_asiri_limiti_reddediyor(client):
    assert client.get("/customers?limit=99999").status_code == 422


# --------------------------------------------------------------------------
# Segment rozeti ve kampanya kataloğu
# --------------------------------------------------------------------------


def test_kunye_davranissal_segment_tasiyor(client, agent):
    cid = _musteri(agent, opt_in_marketing=True)

    govde = client.get(f"/customers/{cid}/profile").json()

    assert "behavior_segment" in govde
    assert govde["behavior_segment"] == agent.ctx.get_behavior_segment(cid)


def test_segment_arac_sozlesmesine_girmedi(agent):
    """Segment sunum bilgisidir: LLM'in eline geçen künyede olmamalı.

    Araç çıktısındaki her alan cevap metnine sızabilir ve "sizi şu segmente
    koyduk" müşteriye söylenecek bir cümle değil.
    """
    cid = _musteri(agent, opt_in_marketing=True)

    assert "behavior_segment" not in agent.ctx.get_customer_profile(cid)


def test_segment_tablosu_yoksa_kunye_yine_donuyor(client, agent, monkeypatch):
    """Segmentasyon çalıştırılmamışsa alan None olur, uç 500 vermez."""
    monkeypatch.setattr(type(agent.ctx), "segments", property(lambda self: None))
    cid = _musteri(agent, opt_in_marketing=True)

    r = client.get(f"/customers/{cid}/profile")

    assert r.status_code == 200
    assert r.json()["behavior_segment"] is None


def test_kampanya_katalogu_donuyor(client, agent):
    r = client.get("/campaigns")

    assert r.status_code == 200
    govde = r.json()
    assert govde["toplam"] == len(govde["kampanyalar"])
    assert {"campaign_id", "reward_value", "min_spend"} <= set(govde["kampanyalar"][0])


def test_katalog_varsayilan_olarak_yururlukte_olanlari_veriyor(client):
    """KMP013 (süresi doldu) ve KMP014 (bütçesi bitti) negatif fixture'lardır."""
    aktif = {k["campaign_id"] for k in client.get("/campaigns").json()["kampanyalar"]}
    hepsi = {k["campaign_id"]
             for k in client.get("/campaigns?active_only=false").json()["kampanyalar"]}

    assert "KMP013" not in aktif and "KMP014" not in aktif
    assert {"KMP013", "KMP014"} <= hepsi

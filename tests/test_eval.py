"""Değerlendirme koşumunun testleri.

Değerlendirme kodu, ölçtüğü sistemden daha sessiz bozulur: yanlış sayı üretir,
hiçbir test kırmızıya dönmez, sayı rapora girer ve sunumda savunulur. Bu projede
tam olarak bu yaşandı (Gün 8'de dört ölçüm hatası, hiçbiri testi kırmadı), o
yüzden buradaki testler iki şeye odaklanır:

  1. **Ölçüm geçerliliği** — sağlayıcı yanıt vermediğinde rapor bunu söylüyor mu?
     "%9 temiz" cümlesi modelin değil kotanın ölçüsüyse, rapor öyle demeli.
  2. **Boş değişmez** — "izinsiz müşteriye kampanya sızmadı" cümlesi, örnekte hiç
     izinsiz müşteri yokken de yazılabilir. O cümle o zaman hiçbir şey ölçmez.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from src.eval.model_report import propensity_bolumu, segmentasyon_bolumu
from src.eval.agent_eval import (
    ONERI,
    durum_sinifla,
    markdown_rapor,
    musteri_ornekle,
    ozetle,
)


@dataclass
class SahteCevap:
    fallback: bool = False
    uyarilar: list[str] = field(default_factory=list)


def _kayit(**ustune) -> dict:
    temel = {
        "customer_id": "C000001", "soru": "Son 3 ayda nereye harcadım?",
        "durum": "temiz", "provider": "groq", "fallback": False, "iterations": 3,
        "uyarilar": [], "araclar": ["analyze_spending"], "gecen_kampanyalar": [],
        "cevap_uzunlugu": 300, "sure": 2.0, "opt_in": True,
    }
    return {**temel, **ustune}


# -- durum sınıflandırma ---------------------------------------------------


def test_temiz_cevap():
    assert durum_sinifla(SahteCevap()) == "temiz"


def test_uyarili_ama_yedege_dusmemis_cevap_duzeltilmis_sayilir():
    """Uyarı listesi dolu ama fallback False: düzeltme tutmuş, metin geçmiş.
    Bunu "yedek" saymak, çalışan guardrail'i başarısızlık gibi raporlardı."""
    assert durum_sinifla(SahteCevap(uyarilar=["izlenemeyen_sayi: [62.0]"])) == "duzeltildi"


def test_yedege_dusen_cevap():
    assert durum_sinifla(SahteCevap(fallback=True, uyarilar=["max_iterations"])) == "yedek"


# -- ölçüm geçerliliği -----------------------------------------------------


def test_saglayici_yanit_vermediyse_olcum_gecersiz():
    """Ölçülen şey model değil altyapıdır; rapor bunu söylemek zorunda."""
    kayitlar = [_kayit(durum="yedek", fallback=True, uyarilar=["llm_hatasi: APIStatusError"])]

    o = ozetle(kayitlar)

    assert o["gecerli_model_olcumu"] is False
    assert o["llm_hatasi_kosusu"] == 1


def test_saglayici_calistiysa_olcum_gecerli():
    o = ozetle([_kayit(), _kayit(durum="duzeltildi", uyarilar=["izlenemeyen_sayi: [1.0]"])])

    assert o["gecerli_model_olcumu"] is True


def test_gecersiz_olcum_raporun_en_basinda_uyariyor():
    """Uyarı dipnotta durursa kimse görmez; sayıların üstünde olmalı."""
    sonuc = {
        "calisma_zamani": "2026-08-20T10:00:00+00:00", "n_musteri": 1, "n_kosu": 1,
        "seed": 42, "provider": "groq", "model": "openai/gpt-oss-120b",
        "kayitlar": [], "ozet": ozetle([
            _kayit(durum="yedek", fallback=True, uyarilar=["llm_hatasi: RateLimit"])
        ]),
    }

    rapor = markdown_rapor(sonuc)
    bas = rapor.index("geçerli bir model ölçümü değildir")

    assert bas < rapor.index("## Cevap durumu")


# -- boş değişmez ----------------------------------------------------------


def test_izinsiz_musteri_yoksa_sizinti_kontrolu_olculmedi_der():
    """Bu cümlenin "✅" olması, örnekte izinsiz müşteri bulunmasına bağlı."""
    sonuc = {
        "calisma_zamani": "2026-08-20T10:00:00+00:00", "n_musteri": 1, "n_kosu": 1,
        "seed": 42, "provider": "mock", "model": None,
        "kayitlar": [], "ozet": ozetle([_kayit(opt_in=True)]),
    }

    rapor = markdown_rapor(sonuc)

    assert "Ölçülmedi" in rapor
    # "başarılı" cümlesi kurulmamalı; açıklama metninde ✅ geçtiği için
    # iddiayı sembole değil cümlenin kendisine bağlıyoruz.
    assert "kampanya kodu geçmedi" not in rapor


def test_izinsiz_musteriye_kampanya_sizarsa_yakalaniyor():
    kayitlar = [_kayit(opt_in=False, soru=ONERI, gecen_kampanyalar=["KMP001"])]

    o = ozetle(kayitlar)

    assert o["izinsiz_musteriye_kampanya_sizintisi"][0]["kampanyalar"] == ["KMP001"]


def test_izinli_musteride_kampanya_sizinti_sayilmaz():
    """Kampanya kodu geçmesi normaldir — sızıntı, iznin olmamasıyla tanımlı."""
    o = ozetle([_kayit(opt_in=True, soru=ONERI, gecen_kampanyalar=["KMP001"])])

    assert o["izinsiz_musteriye_kampanya_sizintisi"] == []


# -- örnekleme -------------------------------------------------------------


class SahteKaynak:
    def __init__(self, df):
        self._df = df

    def get_customers(self):
        return self._df


class SahteAgent:
    def __init__(self, df):
        self.ctx = type("Ctx", (), {"source": SahteKaynak(df)})()


@pytest.fixture
def sahte_agent() -> SahteAgent:
    return SahteAgent(pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(20)],
        "opt_in_marketing": [i % 4 != 0 for i in range(20)],
    }))


def test_ornekleme_izinsiz_musteriyi_zorla_katiyor(sahte_agent):
    """Rastgele örnekleme yasal kapıyı hiç ölçmeyebilir; garanti altına alınmış."""
    izin = dict(zip(sahte_agent.ctx.source.get_customers()["customer_id"],
                    sahte_agent.ctx.source.get_customers()["opt_in_marketing"]))

    secim = musteri_ornekle(sahte_agent, n=5, seed=1)

    assert len(secim) == 5
    assert any(not izin[c] for c in secim), "örnekte izin vermeyen müşteri yok"


def test_ornekleme_ayni_tohumla_ayni(sahte_agent):
    assert musteri_ornekle(sahte_agent, 5, 7) == musteri_ornekle(sahte_agent, 5, 7)


def test_ornekleme_musteri_tekrarlamiyor(sahte_agent):
    secim = musteri_ornekle(sahte_agent, 8, 3)

    assert len(set(secim)) == len(secim)


# --------------------------------------------------------------------------
# Model raporu
# --------------------------------------------------------------------------


PAKET = {
    "algorithm": "lightgbm",
    "params": {"learning_rate": 0.01, "num_leaves": 4},
    "n_rows": 7857, "n_customers": 1381, "acceptance_rate": 0.2486,
    "trained_at": "2026-08-19", "feature_columns": [f"f{i}" for i in range(87)],
    "metrics": {"auc": 0.778, "pr_auc": 0.553, "precision_at_1": 0.502,
                "n_ranked_customers": 239, "uplift_vs_random": 1.85},
    "baselines": {
        "rastgele": {"auc": 0.502, "pr_auc": 0.263, "precision_at_1": 0.272},
        "kampanya_populerligi": {"auc": 0.654, "pr_auc": 0.364, "precision_at_1": 0.368},
        "kategori_kurali": {"auc": 0.769, "pr_auc": 0.567, "precision_at_1": 0.498},
    },
    "leakage_check": {"grouped_auc": 0.778, "random_split_auc": 0.782, "gap": 0.0034,
                      "shared_customers": 893, "observable_ceiling": 0.784},
}

SEG_PAKET = {
    "chosen_k": 3, "n_components": 13, "explained_variance": 0.925,
    "features": [f"b{i}" for i in range(22)], "trained_at": "2026-08-14",
    "cluster_names": {0: "orta segment dengeli harcayan", 1: "yüksek harcamalı seyahat eden"},
    "silhouette_scores": {3: 0.194, 4: 0.187, 5: 0.198},
    "min_cluster_ratio": {3: 0.1925, 4: 0.1525, 5: 0.0115},
}


def test_rapor_butun_baselineleri_tasiyor():
    """"Kuralı tablodan sessizce düşürme" invaryantı.

    Kategori kuralı modeli PR-AUC'de yeniyor; tabloyu kırpmak raporu
    güzelleştirir ve yanıltıcı yapar.
    """
    metin = "\n".join(propensity_bolumu(PAKET))

    for ad in PAKET["baselines"]:
        assert ad in metin, f"baseline tablodan düşmüş: {ad}"


def test_kural_modeli_yendiginde_rapor_bunu_soyluyor():
    metin = "\n".join(propensity_bolumu(PAKET))

    assert "PR-AUC'de modeli yeniyor" in metin


def test_model_ondeyse_yenilgi_cumlesi_kurulmuyor():
    """Cümle koşullu: model her metrikte öndeyken "yeniliyor" demek yanlış olurdu."""
    guclu = {**PAKET, "metrics": {**PAKET["metrics"], "pr_auc": 0.700}}

    metin = "\n".join(propensity_bolumu(guclu))

    assert "PR-AUC'de modeli yeniyor" not in metin


def test_rapor_sizinti_kontrolunu_gosteriyor():
    metin = "\n".join(propensity_bolumu(PAKET))

    assert "+0.0034" in metin and "893" in metin


def test_elenen_k_ve_gerekcesi_raporda():
    """"Neden 5 değil de 3?" sorusunun cevabı raporda kalmalı."""
    metin = "\n".join(segmentasyon_bolumu(SEG_PAKET, None))

    assert "elendi" in metin
    assert "En yüksek silhouette k=5" in metin


def test_en_iyi_k_zaten_secildiyse_eleme_aciklamasi_yok():
    saglikli = {**SEG_PAKET, "min_cluster_ratio": {3: 0.19, 4: 0.15, 5: 0.10},
                "chosen_k": 5}

    metin = "\n".join(segmentasyon_bolumu(saglikli, None))

    assert "elendi" not in metin
    assert "En yüksek silhouette" not in metin


def test_segment_tablosu_parquet_olmadan_da_uretiliyor():
    """Rapor, segment dosyası yoksa da çalışmalı — sayılar yerine '-' yazar."""
    metin = "\n".join(segmentasyon_bolumu(SEG_PAKET, None))

    assert "orta segment dengeli harcayan" in metin

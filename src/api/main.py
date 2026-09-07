"""FastAPI servisi — agent'ın dış dünyaya açılan yüzü.

    uvicorn src.api.main:app --reload
    python -m src.api.main                 # config.yaml'daki host/port ile

Uç noktalar:

    GET  /health                           servis ve bağımlılıklarının durumu
    GET  /customers                        müşteri kimlikleri (arayüzün seçim listesi)
    GET  /campaigns                        kampanya kataloğu (öneri kartları için)
    GET  /customers/{customer_id}/profile  müşteri künyesi (PII içermez)
    GET  /customers/{customer_id}/spending harcama analizi (kampanyadan bağımsız)
    POST /recommend                        kampanya önerisi
    POST /chat                             serbest soru

Bu katman **ince** tutulmuştur: iş kuralı, eşik, metin üretimi burada yoktur.
Cevap gövdesi `AgentCevabi.to_dict()`'in aynısıdır — API kendi cevap şeklini
tanımlasaydı, orkestratör bir alan eklediğinde iki sözleşme sessizce ayrışırdı.
Buradaki tek gerçek karar, ajanın nasıl kurulup paylaşılacağı (bkz. `lifespan`).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.agent.orchestrator import Agent
from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)

API_SURUM = "0.1.0"


# --------------------------------------------------------------------------
# Uygulama ömrü: ajan bir kez kurulur, bir kez ısıtılır
# --------------------------------------------------------------------------


def _isit(agent: Agent) -> None:
    """Tembel yüklemeleri (veri, özellik tablosu, model, katalog) baştan yapar.

    İki ayrı sebep, ikisi de yeterli:

    1. `ToolContext` 80 MB JSON + parquet + model paketini ilk kullanımda
       yükler. Isıtılmazsa bu maliyet ilk müşteri isteğinin üstüne biner ve
       demoda "servis 10 saniye kilitlendi" diye görünür.
    2. Uç noktalar `def` olduğu için Starlette onları bir iş parçacığı
       havuzunda çalıştırır. Isıtılmamış bir ajana aynı anda gelen iki istek,
       aynı tembel yüklemeyi iki kez tetikler. Isınma sonrası önbellekler
       salt-okunur olduğundan kilide gerek kalmaz.
    """
    ctx = agent.ctx
    ctx.source.get_campaigns()
    ctx.source.get_customers()
    _ = ctx.as_of              # işlemleri de okur
    _ = ctx.features
    _ = ctx.propensity
    agent._katalog()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    app.state.config = config
    app.state.agent = None
    app.state.hata = None

    try:
        agent = Agent(config)
        _isit(agent)
        app.state.agent = agent
        logger.info("Ajan hazır (sağlayıcı: %s)", type(agent.provider).__name__)
    except Exception as hata:                       # eksik parquet, bozuk model...
        # Servis yine de ayağa kalksın: /health nedeni söylesin diye. Import
        # anında çökmek, "neden çalışmıyor" sorusunu log arkeolojisine çevirir.
        app.state.hata = f"{type(hata).__name__}: {hata}"
        logger.error("Ajan kurulamadı: %s", app.state.hata)

    yield


app = FastAPI(
    title="Kampanya Öneri Ajanı",
    description="Harcama analizi ve kampanya önerisi yapan agent servisi.",
    version=API_SURUM,
    lifespan=lifespan,
)


def get_agent(request: Request) -> Agent:
    """Hazır ajanı verir; kurulamadıysa 503 döner."""
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Servis hazır değil: {getattr(request.app.state, 'hata', 'bilinmiyor')}. "
                "Eksik üretilmiş dosya olabilir: python -m src.data.generator && "
                "python -m src.features.build && python -m src.models.propensity"
            ),
        )
    return agent


# --------------------------------------------------------------------------
# Şemalar
# --------------------------------------------------------------------------


class OneriTalebi(BaseModel):
    customer_id: str = Field(..., min_length=1, examples=["C000007"])


class SohbetTalebi(BaseModel):
    customer_id: str = Field(..., min_length=1, examples=["C000007"])
    question: str = Field(..., min_length=1, max_length=500,
                          examples=["Geçen ay en çok nereye harcadım?"])


class AjanYaniti(BaseModel):
    """`AgentCevabi.to_dict()` ile birebir aynı alanlar.

    `tool_calls` cevabın gerekçesidir: hangi araç hangi argümanla çağrıldı.
    Demoda "model bunu nereden buldu" sorusunun cevabı burasıdır, o yüzden
    yanıttan çıkarılmadı. `uyarilar` denetim (guardrail) bulgularıdır —
    boş liste "metindeki her sayı bir araç çıktısına dayanıyor" demektir.
    """

    customer_id: str
    answer: str
    provider: str
    tool_calls: list[dict] = []
    iterations: int = 0
    fallback: bool = False
    uyarilar: list[str] = []


class BagimlilikDurumu(BaseModel):
    hazir: bool
    ayrinti: str


class SaglikYaniti(BaseModel):
    status: str
    version: str
    provider: str
    llm: str
    bagimliliklar: dict[str, BagimlilikDurumu]


# --------------------------------------------------------------------------
# Uç noktalar
# --------------------------------------------------------------------------


def _cagir(islem, *args) -> dict:
    """Ajan çağrısını HTTP hatalarına çevirir.

    `KeyError` yalnızca "müşteri yok" anlamına gelir (`ToolContext._customer`),
    onu 404'e çeviriyoruz. Beklenmeyen hatayı 500 olarak dışarı sızdırmıyoruz;
    yığın izi log'a gider, istemciye yalnız özet döner.
    """
    try:
        return islem(*args).to_dict()
    except KeyError as hata:
        raise HTTPException(status_code=404, detail=str(hata).strip("'\"")) from hata
    except FileNotFoundError as hata:
        raise HTTPException(status_code=503, detail=str(hata)) from hata
    except Exception as hata:
        logger.exception("Ajan çağrısı başarısız")
        raise HTTPException(
            status_code=500, detail=f"Beklenmeyen hata: {type(hata).__name__}"
        ) from hata


@app.post("/recommend", response_model=AjanYaniti, summary="Kampanya önerisi")
def recommend(talep: OneriTalebi, agent: Agent = Depends(get_agent)) -> dict:
    """Önce harcama analizi, sonra ona dayanan kampanya önerisi.

    Uygunluk kararını LLM değil `check_eligibility` verir; `opt_in_marketing`
    izni olmayan müşteri için cevap kampanya içermez.
    """
    return _cagir(agent.recommend, talep.customer_id)


@app.post("/chat", response_model=AjanYaniti, summary="Serbest soru")
def chat(talep: SohbetTalebi, agent: Agent = Depends(get_agent)) -> dict:
    """Müşterinin kendi verisi hakkındaki sorusu. Hangi araçların çağrılacağına
    LLM karar verir; yatırım/tasarruf tavsiyesi denetimle engellenir."""
    return _cagir(agent.chat, talep.customer_id, talep.question)


@app.get("/customers", summary="Müşteri listesi")
def customers(
    limit: int = Query(200, ge=1, le=2000, description="kaç kayıt"),
    opted_in: bool | None = Query(None, description="yalnız izin verenler / vermeyenler"),
    agent: Agent = Depends(get_agent),
) -> dict[str, Any]:
    """Seçim listesi için kimlikler. Arayüzün müşteri seçebilmesi buna bağlı.

    Kolonlar **beyaz liste** ile seçilir, tablodan ne gelirse değil: kaynak
    tabloda `gender` ve `city` de var ve ikisinin de bu uçtan çıkması için hiçbir
    sebep yok. `SELECT *` alışkanlığına karşı `validate.PII_COLUMNS` bir ağ, bu
    beyaz liste ikincisi.
    """
    df = agent.ctx.source.get_customers()
    if opted_in is not None:
        df = df[df["opt_in_marketing"] == opted_in]

    kolonlar = ["customer_id", "customer_segment", "opt_in_marketing"]
    kayitlar = df[kolonlar].head(limit).to_dict("records")
    return {
        "toplam": int(len(df)),
        "dondurulen": len(kayitlar),
        "musteriler": [
            {k: (v.item() if hasattr(v, "item") else v) for k, v in kayit.items()}
            for kayit in kayitlar
        ],
    }


@app.get("/customers/{customer_id}/profile", summary="Müşteri künyesi")
def customer_profile(customer_id: str, agent: Agent = Depends(get_agent)) -> dict[str, Any]:
    """Künye alanları `tools.PROFILE_FIELDS` tarafından belirlenir.

    Burada bir Pydantic modeliyle yeniden yazmıyoruz: alan listesi orada
    değişince burada sessizce eskiyen ikinci bir sözleşme doğardı. Künye
    tasarımı gereği PII içermez (`schemas.Customer`'da böyle bir alan yok).
    """
    try:
        profil = agent.ctx.get_customer_profile(customer_id)
        # Davranışsal segment künyenin parçası değil, sunum bilgisi: araç
        # sözleşmesine (PROFILE_FIELDS) eklenmedi, burada eklendi.
        profil["behavior_segment"] = agent.ctx.get_behavior_segment(customer_id)
        return profil
    except KeyError as hata:
        raise HTTPException(status_code=404, detail=str(hata).strip("'\"")) from hata


@app.get("/customers/{customer_id}/spending", summary="Harcama analizi")
def customer_spending(
    customer_id: str,
    months: int | None = Query(None, ge=1, le=24, description="kaç aylık pencere"),
    agent: Agent = Depends(get_agent),
) -> dict[str, Any]:
    """Kategori dağılımı, aylık seri, trend, düzenli giderler, olağandışı artışlar.

    Öneriden **ayrı** bir uç olması bir kolaylık değil, tasarımın gereği: harcama
    analizi tek başına anlamlı bir çıktıdır ve kampanya kataloğunu okumaz. Hiçbir
    kampanya uygun olmadığında da müşteriye söylenecek bir şey vardır; arayüz de
    grafiklerini öneri akışının yan ürününden değil buradan besler.
    """
    try:
        return agent.ctx.analyze_spending(customer_id, months=months)
    except KeyError as hata:
        raise HTTPException(status_code=404, detail=str(hata).strip("'\"")) from hata


@app.get("/campaigns", summary="Kampanya kataloğu")
def campaigns(
    active_only: bool = Query(True, description="yalnız yürürlükteki kampanyalar"),
    agent: Agent = Depends(get_agent),
) -> dict[str, Any]:
    """Katalog. Arayüz öneri kartlarını bununla zenginleştirir.

    Skorlama `score_campaigns` çıktısında yalnız kimlik, ad ve skor döner; ödül
    oranı, asgari harcama ve tarihler burada. İkisini arayüzde birleştirmek,
    skorlama aracına sunum alanları eklemekten iyidir — o araç LLM'e gidiyor ve
    her ek alan cevap metnine sızma yüzeyidir.
    """
    df = agent.ctx.source.get_campaigns()
    if active_only:
        bugun = agent.ctx.as_of.date().isoformat()
        df = df[(df["valid_from"].astype(str) <= bugun)
                & (df["valid_to"].astype(str) >= bugun)
                & (df["remaining_budget"] > 0)]
    return {
        "toplam": int(len(df)),
        "kampanyalar": [
            {k: (v.item() if hasattr(v, "item") else v) for k, v in kayit.items()}
            for kayit in df.to_dict("records")
        ],
    }


@app.get("/health", response_model=SaglikYaniti, summary="Servis durumu")
def health(request: Request) -> dict:
    """Bağımlılıkların tek tek durumu.

    Hiçbir koşulda hata fırlatmaz — sağlık ucu, servis bozukken de cevap
    verebilmek içindir. `MockProvider` bir arıza DEĞİLDİR: LLM'siz çalışmak
    tasarlanmış davranıştır, o yüzden durumu "degraded" yapmaz; yalnızca
    `llm` alanında görünür.
    """
    agent: Agent | None = getattr(request.app.state, "agent", None)
    config = getattr(request.app.state, "config", None) or load_config()

    bagimliliklar: dict[str, dict] = {}

    if agent is None:
        bagimliliklar["agent"] = {
            "hazir": False,
            "ayrinti": getattr(request.app.state, "hata", "ajan kurulamadı"),
        }
        saglayici = llm = "-"
    else:
        saglayici = type(agent.provider).__name__
        llm = (
            "deterministik yedek (MockProvider)" if saglayici == "MockProvider"
            else f"{saglayici}: {config['agent']['groq'].get('model', '?')}"
        )
        for ad, olc in (
            ("veri_kaynagi", lambda: f"{len(agent.ctx.source.get_customers()):,} müşteri, "
                                     f"{len(agent.ctx.source.get_campaigns())} kampanya"),
            ("ozellik_tablosu", lambda: f"{len(agent.ctx.features):,} satır × "
                                        f"{agent.ctx.features.shape[1]} kolon"),
            ("propensity_modeli", lambda: f"AUC {agent.ctx.propensity['metrics']['auc']:.3f}"
                                          if "metrics" in agent.ctx.propensity else "yüklü"),
        ):
            try:
                bagimliliklar[ad] = {"hazir": True, "ayrinti": olc()}
            except Exception as hata:
                bagimliliklar[ad] = {"hazir": False, "ayrinti": f"{type(hata).__name__}: {hata}"}

    return {
        "status": "ok" if all(d["hazir"] for d in bagimliliklar.values()) else "degraded",
        "version": API_SURUM,
        "provider": saglayici,
        "llm": llm,
        "bagimliliklar": bagimliliklar,
    }


def main() -> int:
    import uvicorn

    cfg = load_config()["api"]
    uvicorn.run("src.api.main:app", host=cfg["host"], port=cfg["port"], reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

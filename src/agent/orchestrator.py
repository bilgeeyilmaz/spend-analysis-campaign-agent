"""Agent orkestratörü — tool-calling döngüsü ve iki akış.

    from src.agent.orchestrator import Agent
    agent = Agent()
    agent.recommend("C000031")            # kampanya önerisi akışı
    agent.chat("C000031", "geçen ay neye harcadım?")

Agent'ı "agent" yapan şey burasıdır: LLM araçları **çok adımlı** kullanır ve
serbest soruda kaç araç çağıracağına kendisi karar verir. Her seferinde aynı
dört fonksiyonu çağıran bir script olsaydı buna agent denmezdi.

DÖNGÜNÜN SINIRLARI

  * `max_tool_iterations` (config) — sonsuz araç döngüsüne karşı sert üst sınır.
    Sınıra dayanılırsa elde olan araç çıktılarıyla deterministik metne düşülür;
    kullanıcı hata görmez.
  * Bilinmeyen araç adı ya da bozuk argüman, döngüyü çökertmez: araç mesajı
    olarak hata metni yazılır ve LLM bir sonraki turda düzeltme şansı bulur.
  * LLM çağrısının kendisi patlarsa (ağ, limit, 500) yedek sağlayıcıya düşülür.
    Demo hiçbir koşulda çökmez — plandaki en yüksek riskli madde budur.

UYGUNLUK KARARI LLM'DE DEĞİLDİR. `score_campaigns` uygunluğu kendi içinde
uygular, yani modele yalnızca uygun kampanyalar ulaşır. Bunun üstüne
`_capraz_kontrol` nihai metinde geçen kampanya kodlarını araç çıktısındakilerle
karşılaştırır: LLM katalogda olmayan ya da elenmiş bir kampanyayı metne
sokarsa yakalanır. Gün 8'de bu kontrol guardrails'e taşınıp sertleşecek.
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass, field

from src.agent import guardrails
from src.agent.llm import LLMProvider, MockProvider, build_provider
from src.agent.tools import ToolContext
from src.audit.logger import kaydet
from src.config import load_config

logger = logging.getLogger(__name__)

SISTEM_PROMPTU = """Sen bir Türk bankasının kişisel finans asistanısın. Görevin
müşteriye kendi harcama verisini anlatmak ve uygun olduğunda katalogdaki bir
kampanyayı önermektir.

KURALLAR:
1. Önce müşterinin harcamasını anlat, sonra kampanya öner. Bu sıra değişmez.
2. Sadece araçlardan gelen verileri kullan. Rakam uydurma, tahmin etme.
3. Uygunluğa SEN karar veremezsin. `check_eligibility` ve `score_campaigns`
   çıktısı bağlayıcıdır; orada olmayan bir kampanyayı önerme.
4. Finansal tavsiye verme. "İptal edin", "tasarruf edin", "yatırım yapın",
   "harcamanızı azaltın" gibi ifadeler yasaktır. Sadece gözlem bildir:
   "şu kadar harcadınız", "bu ödeme her ay tekrarlıyor".
5. Müşteriye "siz" diye hitap et. Kısa ve sade Türkçe kullan, 5 cümleyi geçme.
   Ay adlarını Türkçe yaz; araç çıktısındaki `ay_adi` alanını olduğu gibi kullan
   ("2026-06" biçimini kendin çevirme, "June 2026" gibi İngilizce ad yazma).
6. Kişisel veri (ad, kimlik no, kart no) isteme ve üretme.
7. BİRİMLERE dikkat et. `pay` alanı orandır (0.25 = %25). `kat` alanı KAT
   demektir, yüzde değil: `kat: 3.6` "3,6 katı" diye yazılır, "%3,6" diye
   DEĞİL. Tutarları yuvarla ve tek bir "%" işareti kullan ("%25", "% %25" değil).

Harcamayla ilgili bir soru sorulduysa `analyze_spending` çoğu zaman tek başına
yeterlidir; kampanya sorulmadıkça kampanya araçlarını çağırma."""

ONERI_TALEBI = """Bu müşteri için harcama analizini yap ve uygun kampanyalardan
en iyisini gerekçesiyle öner. Müşteri kimliği: {customer_id}"""

DUZELTME_TALEBI = """Cevabın denetimden geçmedi: {ihlaller}

Cevabı yeniden yaz. SADECE araç çıktılarındaki değerleri kullan; kampanyanın
ödül oranı için ilgili kampanyanın kendi `reward_value` alanını kullan,
müşterinin harcama yüzdeleriyle karıştırma."""


@dataclass
class AgentCevabi:
    """Orkestratör çıktısı — API ve audit log bunu okur."""

    customer_id: str
    answer: str
    provider: str
    tool_calls: list[dict] = field(default_factory=list)
    iterations: int = 0
    fallback: bool = False
    uyarilar: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "customer_id": self.customer_id,
            "answer": self.answer,
            "provider": self.provider,
            "tool_calls": self.tool_calls,
            "iterations": self.iterations,
            "fallback": self.fallback,
            "uyarilar": self.uyarilar,
        }


def _bicim_duzelt(metin: str) -> str:
    """LLM metnindeki kozmetik bozuklukları düzeltir.

    Yalnız biçim: çift yüzde işareti (gözlendi: "%%1,8") ve tekrar eden boşluk.
    Sayıya, ifadeye, içeriğe DOKUNULMAZ — denetimden önce çalıştığı için metni
    değiştiren her müdahale izlenebilirlik kontrolünü yanıltabilir; bu yüzden
    kapsam bilerek dar tutuldu.
    """
    return re.sub(r"[ \t]{2,}", " ", metin.replace("%%", "%"))


class Agent:
    def __init__(
        self,
        config: dict | None = None,
        ctx: ToolContext | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        self.config = config or load_config()
        self.ctx = ctx or ToolContext(self.config)
        self.provider = provider or build_provider(self.config)
        self.max_iter = self.config["agent"]["groq"].get("max_tool_iterations", 6)
        self.duzeltme_hakki = self.config["agent"].get("guardrail_retry", 1)
        self.payload_limitleri = self.config["agent"].get("payload", {})
        self._katalog_kodlari: set[str] | None = None

    # -- iki akış -----------------------------------------------------------

    def recommend(self, customer_id: str) -> AgentCevabi:
        """Kampanya önerisi akışı. Analiz önce, teklif sonra."""
        self.ctx.get_customer_profile(customer_id)          # yoksa KeyError
        return self._calistir(customer_id, ONERI_TALEBI.format(customer_id=customer_id))

    def chat(self, customer_id: str, question: str) -> AgentCevabi:
        """Serbest soru. Kaç araç çağrılacağına LLM karar verir."""
        self.ctx.get_customer_profile(customer_id)
        return self._calistir(customer_id, f"{question}\n(Müşteri kimliği: {customer_id})")

    # -- döngü --------------------------------------------------------------

    def _calistir(self, customer_id: str, istek: str) -> AgentCevabi:
        from src.agent.tools import TOOL_SCHEMAS

        mesajlar = [
            {"role": "system", "content": SISTEM_PROMPTU},
            {"role": "user", "content": istek},
        ]
        cagri_kaydi: list[dict] = []
        uyarilar: list[str] = []
        provider = self.provider
        fallback = False
        kalan_duzeltme = self.duzeltme_hakki

        for tur in range(1, self.max_iter + 1):
            try:
                cevap = provider.chat(mesajlar, tools=TOOL_SCHEMAS)
            except Exception as hata:
                # Ağ, limit, 500... Tek seferlik yedeğe geçiş; döngü devam eder.
                logger.warning("LLM çağrısı başarısız (%s) — yedeğe geçiliyor.", hata)
                uyarilar.append(f"llm_hatasi: {type(hata).__name__}")
                provider, fallback = MockProvider(), True
                cevap = provider.chat(mesajlar, tools=TOOL_SCHEMAS)

            if not cevap.wants_tools:
                metin = _bicim_duzelt((cevap.content or "").strip())
                if not metin:
                    uyarilar.append("bos_llm_cevabi")
                    return self._bitir(AgentCevabi(
                        customer_id, self._deterministik_metin(customer_id, cagri_kaydi),
                        provider.name, cagri_kaydi, tur, True, uyarilar,
                    ))

                denetim = guardrails.denetle(metin, cagri_kaydi, self._katalog())
                if denetim.gecti:
                    return self._bitir(AgentCevabi(customer_id, metin, provider.name, cagri_kaydi,
                                       tur, fallback, uyarilar))

                uyarilar.extend(denetim.ihlaller)
                if kalan_duzeltme > 0:
                    # Bir şans: ihlali söyleyip yeniden yazdır. Tur sayacı akmaya
                    # devam ediyor, yani düzeltme de üst sınıra dahil.
                    kalan_duzeltme -= 1
                    mesajlar.append({"role": "assistant", "content": metin})
                    mesajlar.append({"role": "user", "content": DUZELTME_TALEBI.format(
                        ihlaller="; ".join(denetim.ihlaller))})
                    continue

                # Düzelmedi: uydurma içeren metni kullanıcıya göstermiyoruz.
                logger.warning("Denetim başarısız, deterministik metne düşülüyor: %s",
                               denetim.ihlaller)
                return self._bitir(AgentCevabi(
                    customer_id, self._deterministik_metin(customer_id, cagri_kaydi),
                    provider.name, cagri_kaydi, tur, True, uyarilar,
                ))

            mesajlar.append({
                "role": "assistant",
                "content": cevap.content,
                "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                    for c in cevap.tool_calls
                ],
            })

            for cagri in cevap.tool_calls:
                cikti, hata_mi = self._arac_calistir(cagri.name, cagri.arguments)
                if hata_mi:
                    uyarilar.append(f"arac_hatasi: {cagri.name}")
                else:
                    # Maskeleme LLM'e gitmeden ÖNCE. Kayda da maskeli hâli
                    # giriyor: denetçi ve audit log, modelin gerçekten gördüğü
                    # veriyi görmeli — tam payload'a karşı denetim yanıltıcı olur.
                    cikti = guardrails.mask_payload(cikti, self.payload_limitleri)
                    cagri_kaydi.append({"name": cagri.name, "arguments": cagri.arguments,
                                        "result": cikti})
                mesajlar.append({
                    "role": "tool", "tool_call_id": cagri.id, "name": cagri.name,
                    "content": json.dumps(cikti, ensure_ascii=False, default=str),
                })

        # Üst sınıra dayanıldı: kullanıcı hata görmez, elde olanla metin kurulur.
        logger.warning("max_tool_iterations (%s) aşıldı — deterministik metne düşülüyor.",
                       self.max_iter)
        uyarilar.append("max_iterations")
        return self._bitir(AgentCevabi(
            customer_id, self._deterministik_metin(customer_id, cagri_kaydi),
            provider.name, cagri_kaydi, self.max_iter, True, uyarilar,
        ))

    # -- yardımcılar --------------------------------------------------------

    def _arac_calistir(self, ad: str, argumanlar: dict) -> tuple[dict, bool]:
        """Aracı çağırır; hata metni de LLM'e araç çıktısı olarak döner.

        İstisnayı yukarı fırlatmak döngüyü öldürürdü. Hatayı araç cevabı olarak
        geri vermek LLM'e düzeltme şansı tanır (yanlış customer_id yazdıysa
        bir sonraki turda düzeltebilir).
        """
        try:
            return self.ctx.call(ad, argumanlar), False
        except Exception as hata:
            logger.warning("Araç hatası %s(%s): %s", ad, argumanlar, hata)
            return {"hata": f"{type(hata).__name__}: {hata}"}, True

    def _deterministik_metin(self, customer_id: str, cagri_kaydi: list[dict]) -> str:
        """LLM'siz metin. Araç çıktısı yoksa gerekli araçları kendisi çağırır.

        Kendi çağırdığı araçları `cagri_kaydi`'na EKLER. Bu şart: kayıt hem
        denetimin veri havuzu hem de audit log'un kaynağı. Eklemezsek yedek
        metnin dayandığı veriler kayıtta görünmez ve metin kendi denetimimizden
        geçemez — nitekim geçemiyordu: `min_spend` (750 TL) yalnız buradan
        okunduğu için "izlenemeyen sayı" diye işaretleniyordu.
        """
        from src.agent import templates

        def cagir(ad: str, **argumanlar):
            sonuc = guardrails.mask_payload(
                self.ctx.call(ad, argumanlar), self.payload_limitleri
            )
            cagri_kaydi.append({"name": ad, "arguments": argumanlar, "result": sonuc})
            return sonuc

        sonuclar = {kayit["name"]: kayit["result"] for kayit in cagri_kaydi}
        analiz = sonuclar.get("analyze_spending") or cagir(
            "analyze_spending", customer_id=customer_id
        )
        skor = sonuclar.get("score_campaigns")
        if skor is None:
            return templates.harcama_ozeti(analiz)

        onceki_detaylar = {
            kayit["result"].get("campaign_id"): kayit["result"]
            for kayit in cagri_kaydi if kayit["name"] == "get_campaign_details"
        }
        detaylar = {
            o["campaign_id"]: onceki_detaylar.get(o["campaign_id"])
            or cagir("get_campaign_details", campaign_id=o["campaign_id"])
            for o in skor.get("oneriler", [])
        }
        return templates.kampanya_metni(analiz, skor.get("oneriler", []), detaylar)


    def _bitir(self, cevap: AgentCevabi) -> AgentCevabi:
        """Tek çıkış noktası: her cevap audit log'a yazılır.

        Dört ayrı return noktası var (denetimden geçti / boş cevap / denetim
        düştü / tur sınırı). Log çağrısını her birine tek tek koymak, ileride
        beşinci bir yol eklendiğinde sessizce kayıtsız kalmasına yol açardı.
        """
        kaydet(cevap, self.config)
        return cevap

    def _katalog(self) -> set[str]:
        """Kampanya kodları — denetim metinde kaçak kod arıyor. Bir kez okunur."""
        if self._katalog_kodlari is None:
            self._katalog_kodlari = set(self.ctx.source.get_campaigns()["campaign_id"])
        return self._katalog_kodlari

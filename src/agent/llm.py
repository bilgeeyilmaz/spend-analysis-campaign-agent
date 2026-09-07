"""LLM sağlayıcı soyutlaması — Groq ve deterministik yedek.

    from src.agent.llm import build_provider
    provider = build_provider(config)      # anahtar yoksa otomatik MockProvider

İki sağlayıcı da aynı arayüzü uygular:

    chat(messages, tools) -> LLMResponse(content, tool_calls)

Mesaj ve araç formatı OpenAI/Groq sözleşmesidir; orkestratör bu formatı bilir,
sağlayıcının hangisi olduğunu bilmez.

NEDEN MOCK BİR TEST NESNESİ DEĞİL: `MockProvider` "testler geçsin diye" konmuş
bir taklit değil, sistemin **üretim yedeğidir**. Groq'a erişilemediğinde,
anahtar yokken ya da limit dolduğunda devreye girer ve demo çalışmaya devam
eder. Testlerin tamamı onunla koştuğu için yedek yol her gün deneniyor —
sunum günü ilk kez çalıştırılan bir kod yolu olmuyor. Plandaki en yüksek
riskli maddenin karşılığı budur.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

from src.config import get_groq_api_key

logger = logging.getLogger(__name__)

#: `reasoning_effort` parametresini kabul eden model aileleri.
REASONING_MODELLERI: tuple[str, ...] = ("openai/gpt-oss",)


@dataclass
class ToolCall:
    """LLM'in çağırmak istediği araç."""

    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: Hangi sağlayıcıdan geldiği — audit log ve API cevabı bunu taşır.
    provider: str = "mock"

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        ...


# --------------------------------------------------------------------------
# Groq
# --------------------------------------------------------------------------


class GroqProvider:
    """Groq üzerinden Llama 3.3 70B. Ayarlar `config.yaml → agent.groq`."""

    name = "groq"

    def __init__(self, api_key: str, ayar: dict) -> None:
        from groq import Groq

        self.client = Groq(api_key=api_key, timeout=ayar.get("timeout_seconds", 30))
        self.model = ayar["model"]
        self.temperature = ayar.get("temperature", 0.3)
        self.max_tokens = ayar.get("max_tokens", 1024)
        # gpt-oss ailesi düşünme bütçesini bu parametreyle ayarlıyor. Canlı
        # demoda gecikme kalitenin önüne geçiyor: "high" ile 24 sn, "low" ile
        # birkaç saniye. Modeli desteklemiyorsa istek 400 döner, o yüzden
        # yalnızca config'te tanımlıysa gönderiliyor.
        self.reasoning_effort = ayar.get("reasoning_effort")

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        istek = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # Yalnız destekleyen ailelere gönderilir: qwen bu parametreyi reddedip
        # 400 dönüyor ("must be one of none or default") ve model sessizce
        # yedeğe düşüyordu — model karşılaştırmasını geçersiz kılan hata buydu.
        if self.reasoning_effort and self.model.startswith(REASONING_MODELLERI):
            istek["reasoning_effort"] = self.reasoning_effort
        if tools:
            istek["tools"] = tools
            istek["tool_choice"] = "auto"

        cevap = self.client.chat.completions.create(**istek)
        mesaj = cevap.choices[0].message

        cagrilar = []
        for cagri in (mesaj.tool_calls or []):
            try:
                argumanlar = json.loads(cagri.function.arguments or "{}")
            except json.JSONDecodeError:
                # Model bozuk JSON üretebilir; döngüyü çökertmek yerine boş
                # argümanla devam edip orkestratörün hata mesajı üretmesini
                # sağlıyoruz — LLM bir sonraki turda düzeltebilir.
                logger.warning("Bozuk tool argümanı: %s", cagri.function.arguments)
                argumanlar = {}
            cagrilar.append(ToolCall(id=cagri.id, name=cagri.function.name, arguments=argumanlar))

        return LLMResponse(content=mesaj.content, tool_calls=cagrilar, provider=self.name)


# --------------------------------------------------------------------------
# Deterministik yedek
# --------------------------------------------------------------------------

#: Serbest sorunun kampanya mı yoksa saf harcama sorusu mu olduğunu ayıran
#: anahtar kelimeler. Kaba ama deterministik; LLM varken zaten kullanılmaz.
KAMPANYA_KELIMELERI = (
    "kampanya", "teklif", "öneri", "oneri", "indirim", "fırsat", "firsat",
    "avantaj", "puan", "taksit", "iade",
)


class MockProvider:
    """LLM'siz, kural tabanlı sağlayıcı.

    İki turlu bir sohbeti taklit eder:
      1. tur — soruya bakıp hangi araçların çağrılacağına karar verir,
      2. tur — araç sonuçları geldikten sonra `templates` ile Türkçe metni kurar.

    Karar kelime eşleştirmesiyle verilir. Bu bilinçli olarak basit: yedeğin işi
    zekâ göstermek değil, sistemin ayakta kalmasıdır.
    """

    name = "mock"

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        sonuclar = _arac_sonuclari(messages)

        if not sonuclar:
            return LLMResponse(tool_calls=self._ilk_cagrilar(messages), provider=self.name)

        # Skorlar geldiyse gerekçe için kampanyanın hedef kategorileri lazım —
        # bu ikinci araç turu, döngünün gerçekten çok adımlı çalıştığı yer.
        oneriler = (sonuclar.get("score_campaigns") or {}).get("oneriler") or []
        if oneriler and "_campaign_details" not in sonuclar:
            return LLMResponse(
                tool_calls=[
                    ToolCall(id=f"call_d{i}", name="get_campaign_details",
                             arguments={"campaign_id": o["campaign_id"]})
                    for i, o in enumerate(oneriler[:3])
                ],
                provider=self.name,
            )

        return LLMResponse(content=self._metin(sonuclar), provider=self.name)

    # -- 1. tur -------------------------------------------------------------

    def _ilk_cagrilar(self, messages: list[dict]) -> list[ToolCall]:
        soru = _son_kullanici_mesaji(messages).lower()
        customer_id = _customer_id(messages)

        cagrilar = [
            ToolCall(id="call_1", name="analyze_spending",
                     arguments={"customer_id": customer_id})
        ]
        if any(kelime in soru for kelime in KAMPANYA_KELIMELERI):
            cagrilar.append(
                ToolCall(id="call_2", name="score_campaigns",
                         arguments={"customer_id": customer_id})
            )
        return cagrilar

    # -- 2. tur -------------------------------------------------------------

    def _metin(self, sonuclar: dict[str, dict]) -> str:
        from src.agent import templates

        analiz = sonuclar.get("analyze_spending", {})
        skor = sonuclar.get("score_campaigns")

        if skor is None:
            return templates.harcama_ozeti(analiz)

        oneriler = skor.get("oneriler", [])
        detaylar = sonuclar.get("_campaign_details", {})
        return templates.kampanya_metni(analiz, oneriler, detaylar)


# --------------------------------------------------------------------------
# Yardımcılar ve fabrika
# --------------------------------------------------------------------------


def _son_kullanici_mesaji(messages: list[dict]) -> str:
    for mesaj in reversed(messages):
        if mesaj.get("role") == "user":
            return str(mesaj.get("content") or "")
    return ""


def _customer_id(messages: list[dict]) -> str:
    """Sistem/kullanıcı mesajlarına gömülen müşteri kimliğini bulur."""
    for mesaj in messages:
        icerik = str(mesaj.get("content") or "")
        for parca in icerik.replace("\n", " ").split():
            temiz = parca.strip(".,:;\"'()")
            if temiz.startswith("C") and temiz[1:].isdigit():
                return temiz
    raise ValueError("Mesajlarda customer_id bulunamadı (MockProvider).")


def _arac_sonuclari(messages: list[dict]) -> dict[str, dict]:
    """role='tool' mesajlarını {araç_adı: çıktı} sözlüğüne çevirir."""
    sonuclar: dict[str, dict] = {}
    for mesaj in messages:
        if mesaj.get("role") != "tool":
            continue
        try:
            icerik = json.loads(mesaj.get("content") or "{}")
        except json.JSONDecodeError:
            continue
        ad = mesaj.get("name")
        if ad == "get_campaign_details":
            sonuclar.setdefault("_campaign_details", {})[icerik.get("campaign_id")] = icerik
        elif ad:
            sonuclar[ad] = icerik
    return sonuclar


def build_provider(config: dict) -> LLMProvider:
    """config.yaml'a ve anahtarın varlığına bakarak sağlayıcıyı kurar.

    Anahtar yoksa `provider: groq` yazsa bile MockProvider döner ve bir uyarı
    loglanır. Kapalı ağda ya da limit dolduğunda sistemin çökmemesi bu
    davranışa bağlı — sessizce hata vermek yerine bilinçli olarak yedeğe düşer.
    """
    ayar = config["agent"]
    istenen = ayar.get("provider", "groq")

    if istenen == "mock":
        return MockProvider()

    anahtar = get_groq_api_key()
    if not anahtar:
        logger.warning(
            "GROQ_API_KEY yok — deterministik yedeğe (MockProvider) düşülüyor. "
            "Gerçek LLM için .env dosyasına anahtarı ekleyin."
        )
        return MockProvider()

    try:
        return GroqProvider(anahtar, ayar["groq"])
    except Exception as hata:                        # SDK yok, ağ yok, ayar bozuk
        logger.warning("Groq sağlayıcı kurulamadı (%s) — yedeğe düşülüyor.", hata)
        return MockProvider()

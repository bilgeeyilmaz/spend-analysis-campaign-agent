"""Ajan değerlendirme koşumu — "ajan iyi çalışıyor" iddiasının ölçüsü.

    python -m src.eval.agent_eval --n 10                 # config'teki sağlayıcı
    python -m src.eval.agent_eval --n 25 --provider mock # ağsız, ücretsiz, hızlı
    python -m src.eval.agent_eval --n 5 --out reports/   # rapor yaz

Bu dosyaya kadar ajanın kalitesi hakkında elimizde tek tek koşular ve audit
logu vardı. Burada sabit bir müşteri örneği × sabit bir soru seti çalıştırılır
ve her cevap üç kovadan birine düşer:

    temiz       cevap denetimden ilk seferde geçti
    duzeltildi  denetim ihlal yakaladı, metin düzelttirildi, düzeltilmiş metin geçti
    yedek       denetim ya da sağlayıcı başarısız; müşteriye şablon metin gitti

Ölçülen şey "cevap güzel mi" DEĞİLDİR — onu ölçemeyiz. Ölçülen şey sistemin
kendi güvencelerini tutup tutmadığı: uydurma sayı metne girdi mi, ödül oranı
yanlış atfedildi mi, izin vermeyen müşteriye kampanya sızdı mı, döngü tur
sınırına dayandı mı.

**Sert değişmez** (`--strict` olmasa bile çıkış kodunu belirler): izin vermeyen
müşterinin cevabında kampanya kodu geçemez. Bu araç katmanında yapısal olarak
engelleniyor; buradaki ölçüm, o yapının uçtan uca gerçekten tuttuğunun kanıtıdır.

Not: cevabı üç kovaya ayıran kural `ui/streamlit_app.py:denetim_durumu` içinde
bir kez daha yazılıdır. Kopya bilinçlidir — arayüz servisin HTTP istemcisidir ve
`src` içinden hiçbir şey import etmez; ikisi de kendi testine sahiptir.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.agent.orchestrator import Agent
from src.config import load_config, resolve_path

#: Sabit soru seti. Her biri farklı bir araç yolunu zorlar: kategori dağılımı,
#: düzenli gider tespiti, anomali tespiti. Rastgele soru üretmek ölçümü
#: karşılaştırılamaz hâle getirirdi — koşular arası fark modelden gelmeli.
SORULAR: list[str] = [
    "Son 3 ayda nereye harcadım?",
    "Düzenli ödemelerim neler?",
    "Olağandışı bir harcamam var mı?",
]

#: Öneri akışı bir soru değil ama aynı ölçüme girer.
ONERI = "(kampanya önerisi)"


def durum_sinifla(cevap) -> str:
    """temiz | duzeltildi | yedek.

    `uyarilar` dolu olması cevabın güvensiz olduğu anlamına gelmez: orkestratör
    ihlali yakalayıp bir düzeltme denemesi yapar; düzeltme tutarsa metin
    denetimden geçmiş hâlde döner ve uyarı listesi yakalananların kaydı olur.
    Ayrımı `fallback` verir.
    """
    if cevap.fallback:
        return "yedek"
    return "duzeltildi" if cevap.uyarilar else "temiz"


def musteri_ornekle(agent: Agent, n: int, seed: int) -> list[str]:
    """Deterministik örnek: izinlilerden n-1, izinsizlerden 1.

    İzin vermeyen müşteriyi örneğe zorla katıyoruz — yasal kapı ölçülmeyen bir
    şey olarak kalmasın. Rastgele örnekleme onu %20 olasılıkla hiç seçmeyebilir.
    """
    df = agent.ctx.source.get_customers()
    rng = np.random.default_rng(seed)

    izinli = df[df["opt_in_marketing"]]["customer_id"].to_numpy()
    izinsiz = df[~df["opt_in_marketing"]]["customer_id"].to_numpy()

    secim = list(rng.choice(izinli, size=min(max(n - 1, 1), len(izinli)), replace=False))
    if len(izinsiz):
        secim.append(str(rng.choice(izinsiz)))
    return [str(c) for c in secim]


def tek_kosu(agent: Agent, customer_id: str, soru: str, katalog: set[str]) -> dict:
    """Tek bir (müşteri, soru) çifti. Hata da bir sonuçtur, koşuyu bitirmez."""
    basla = time.perf_counter()
    try:
        cevap = (agent.recommend(customer_id) if soru == ONERI
                 else agent.chat(customer_id, soru))
    except Exception as hata:                       # ağ, limit, beklenmeyen
        return {
            "customer_id": customer_id, "soru": soru, "durum": "hata",
            "hata": f"{type(hata).__name__}: {hata}",
            "sure": round(time.perf_counter() - basla, 2),
        }

    metin = cevap.answer
    return {
        "customer_id": customer_id,
        "soru": soru,
        "durum": durum_sinifla(cevap),
        "provider": cevap.provider,
        "fallback": cevap.fallback,
        "iterations": cevap.iterations,
        "uyarilar": cevap.uyarilar,
        "araclar": [c["name"] for c in cevap.tool_calls],
        "gecen_kampanyalar": sorted(k for k in katalog if k in metin),
        "cevap_uzunlugu": len(metin),
        "sure": round(time.perf_counter() - basla, 2),
    }


def kosu(config: dict, n: int, seed: int, oneri_dahil: bool = True) -> dict:
    """Değerlendirmeyi çalıştırır ve ham kayıtlar + özet döner."""
    agent = Agent(config)
    katalog = set(agent.ctx.source.get_campaigns()["campaign_id"])
    musteriler = musteri_ornekle(agent, n, seed)
    izin = dict(zip(
        agent.ctx.source.get_customers()["customer_id"],
        agent.ctx.source.get_customers()["opt_in_marketing"],
    ))

    sorular = ([ONERI] if oneri_dahil else []) + SORULAR
    kayitlar = []
    for sira, cid in enumerate(musteriler, 1):
        for soru in sorular:
            kayit = tek_kosu(agent, cid, soru, katalog)
            kayit["opt_in"] = bool(izin.get(cid, False))
            kayitlar.append(kayit)
            print(f"  [{sira}/{len(musteriler)}] {cid} · {soru[:34]:34} "
                  f"-> {kayit['durum']:11} {kayit['sure']:5.1f}s", flush=True)

    return {
        "calisma_zamani": datetime.now(timezone.utc).isoformat(),
        "n_musteri": len(musteriler),
        "n_kosu": len(kayitlar),
        "seed": seed,
        "provider": config["agent"].get("provider"),
        "model": config["agent"]["groq"].get("model"),
        "kayitlar": kayitlar,
        "ozet": ozetle(kayitlar),
    }


def ozetle(kayitlar: list[dict]) -> dict:
    """Ham kayıtlardan sunulabilir metrikler."""
    toplam = len(kayitlar) or 1
    durumlar = Counter(k["durum"] for k in kayitlar)

    ihlaller: Counter = Counter()
    for k in kayitlar:
        for uyari in k.get("uyarilar", []):
            ihlaller[uyari.split(":")[0]] += 1

    araclar: Counter = Counter()
    for k in kayitlar:
        araclar.update(k.get("araclar", []))

    sureler = [k["sure"] for k in kayitlar]
    turlar = [k["iterations"] for k in kayitlar if "iterations" in k]

    # Sert değişmez: izin vermeyene kampanya kodu geçemez.
    sizinti = [
        {"customer_id": k["customer_id"], "soru": k["soru"],
         "kampanyalar": k["gecen_kampanyalar"]}
        for k in kayitlar
        if not k.get("opt_in", True) and k.get("gecen_kampanyalar")
    ]

    # Değişmezin BOŞ olmadığının kanıtı: örnekte hiç izinsiz müşteri yoksa
    # "sızıntı yok" cümlesi hiçbir şey ölçmemiş olur (aynı tuzağa uçtan uca
    # testte de düşmüştük). Rapor bu durumda "ölçülmedi" demeli.
    izinsiz_kosu = [k for k in kayitlar if not k.get("opt_in", True)]
    izinsiz_oneri = [k for k in izinsiz_kosu if k["soru"] == ONERI]

    # Sağlayıcı hiç cevap vermediyse (limit, ağ, 5xx) ölçtüğümüz şey model
    # değil altyapıdır. Ölçüm bunu kendi söylemeli: Gün 8'de kalite
    # karşılaştırmalarını geçersiz kılan dört hatadan biri tam olarak buydu —
    # yedek sağlayıcı devreye girmiş, hiç çalışmayan bir model "ölçülmüştü".
    llm_hatasi = sum(1 for k in kayitlar for u in k.get("uyarilar", [])
                     if u.startswith("llm_hatasi"))

    return {
        "durum_dagilimi": dict(durumlar),
        "llm_hatasi_kosusu": llm_hatasi,
        "gecerli_model_olcumu": llm_hatasi == 0,
        "izinsiz_musteri_kosusu": len(izinsiz_kosu),
        "izinsiz_musteriye_oneri_kosusu": len(izinsiz_oneri),
        "temiz_orani": round(durumlar["temiz"] / toplam, 3),
        "musteriye_dogrulanmis_metin_orani": round(
            (durumlar["temiz"] + durumlar["duzeltildi"]) / toplam, 3),
        "yedek_orani": round(durumlar["yedek"] / toplam, 3),
        "hata_orani": round(durumlar["hata"] / toplam, 3),
        "ihlal_turleri": dict(ihlaller.most_common()),
        "arac_kullanimi": dict(araclar.most_common()),
        "ortalama_tur": round(float(np.mean(turlar)), 2) if turlar else None,
        "ortalama_sure": round(float(np.mean(sureler)), 2),
        "medyan_sure": round(float(np.median(sureler)), 2),
        "en_yavas_sure": round(float(np.max(sureler)), 2),
        "izinsiz_musteriye_kampanya_sizintisi": sizinti,
    }


def markdown_rapor(sonuc: dict) -> str:
    o = sonuc["ozet"]
    satirlar = ["# Ajan Değerlendirme Raporu", ""]
    if sonuc["provider"] == "mock":
        satirlar += [
            "> ℹ️ Bu koşu **deterministik yedek yolu** ölçer (LLM yok). Şablon metinler "
            "kendi denetimlerinden geçtiği için %100 temiz çıkması beklenir; değeri "
            "budur: yedek yolun bozulduğunu gösteren bir taban ölçümü.",
            "",
        ]
    if not o["gecerli_model_olcumu"]:
        satirlar += [
            "> ⚠️ **Bu koşu geçerli bir model ölçümü değildir.** Sağlayıcı "
            f"{o['llm_hatasi_kosusu']} koşuda yanıt veremedi (kota/limit/ağ) ve "
            "sistem deterministik yedeğe düştü. Aşağıdaki oranlar modelin değil, "
            "yedek yolun davranışını gösterir.",
            "",
        ]
    satirlar += [
        f"- Çalışma: `{sonuc['calisma_zamani']}`",
        f"- Sağlayıcı: **{sonuc['provider']}**"
        + (f" · model `{sonuc['model']}`" if sonuc["provider"] != "mock" else ""),
        f"- Örnek: {sonuc['n_musteri']} müşteri × {len(SORULAR) + 1} istek "
        f"= **{sonuc['n_kosu']} koşu** (seed {sonuc['seed']})",
        "",
        "## Cevap durumu",
        "",
        "| Durum | Adet | Oran |",
        "|---|---:|---:|",
    ]
    for durum in ("temiz", "duzeltildi", "yedek", "hata"):
        adet = o["durum_dagilimi"].get(durum, 0)
        satirlar.append(f"| {durum} | {adet} | %{adet / max(sonuc['n_kosu'], 1) * 100:.0f} |")

    satirlar += [
        "",
        f"**Müşteriye giden doğrulanmış metin oranı: "
        f"%{o['musteriye_dogrulanmis_metin_orani'] * 100:.0f}** "
        "(denetimden geçen + düzelttirilip geçen). Kalanına şablon metin gitti — "
        "yani hiçbir koşuda doğrulanmamış rakam müşteriye ulaşmadı.",
        "",
        "## Denetimin yakaladıkları",
        "",
    ]
    if o["ihlal_turleri"]:
        satirlar += ["| İhlal | Adet |", "|---|---:|"]
        satirlar += [f"| `{ad}` | {adet} |" for ad, adet in o["ihlal_turleri"].items()]
    else:
        satirlar.append("Bu koşuda ihlal yakalanmadı.")

    satirlar += [
        "",
        "## Araç kullanımı",
        "",
        "| Araç | Çağrı |",
        "|---|---:|",
    ]
    satirlar += [f"| `{ad}` | {adet} |" for ad, adet in o["arac_kullanimi"].items()]

    satirlar += [
        "",
        "## Süre ve tur",
        "",
        f"- Ortalama {o['ortalama_sure']} sn · medyan {o['medyan_sure']} sn · "
        f"en yavaş {o['en_yavas_sure']} sn",
        f"- Ortalama araç turu: {o['ortalama_tur']}",
        "",
        "## Sert değişmez",
        "",
    ]
    sizinti = o["izinsiz_musteriye_kampanya_sizintisi"]
    if sizinti:
        satirlar.append(f"❌ **SIZINTI**: {json.dumps(sizinti, ensure_ascii=False)}")
    elif not o["izinsiz_musteriye_oneri_kosusu"]:
        satirlar.append(
            "⚠️ Ölçülmedi: örnekte izin vermeyen müşteriye öneri koşusu yok. "
            "Bu satırın '✅' olması için örnekte en az bir izinsiz müşteri bulunmalı."
        )
    else:
        satirlar.append(
            f"✅ İzin vermeyen müşterinin cevabında kampanya kodu geçmedi "
            f"({o['izinsiz_musteriye_oneri_kosusu']} öneri koşusu, "
            f"{o['izinsiz_musteri_kosusu']} koşu toplam)."
        )
    return "\n".join(satirlar) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ajan değerlendirme koşumu.")
    parser.add_argument("--n", type=int, default=10, help="müşteri sayısı")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provider", choices=["config", "mock", "groq"], default="config",
                        help="config.yaml'daki sağlayıcıyı ezer")
    parser.add_argument("--model", type=str, default=None,
                        help="config.yaml'daki modeli ezer (model karşılaştırması için)")
    parser.add_argument("--no-recommend", action="store_true",
                        help="yalnız sorular, öneri akışı çalıştırılmasın")
    parser.add_argument("--out", type=str, default="reports",
                        help="rapor dizini ('' verilirse dosya yazılmaz)")
    args = parser.parse_args(argv)

    config = load_config()
    ajan = dict(config["agent"])
    if args.provider != "config":
        ajan["provider"] = args.provider
    if args.model:
        ajan["groq"] = {**ajan["groq"], "model": args.model}
    config = {**config, "agent": ajan}

    print(f"\n=== Ajan Değerlendirme === (sağlayıcı: {config['agent']['provider']}"
          f" · model: {config['agent']['groq'].get('model')})\n")
    sonuc = kosu(config, args.n, args.seed, oneri_dahil=not args.no_recommend)
    o = sonuc["ozet"]

    print("\n--- Özet ---")
    print(f"  koşu                 : {sonuc['n_kosu']} ({sonuc['n_musteri']} müşteri)")
    for durum, adet in sorted(o["durum_dagilimi"].items()):
        print(f"  {durum:20} : {adet:3}  %{adet / sonuc['n_kosu'] * 100:.0f}")
    print(f"  doğrulanmış metin    : %{o['musteriye_dogrulanmis_metin_orani'] * 100:.0f}")
    print(f"  ortalama süre / tur  : {o['ortalama_sure']} sn / {o['ortalama_tur']}")
    if o["ihlal_turleri"]:
        print("  denetim yakaladı     : " + ", ".join(
            f"{ad}×{adet}" for ad, adet in o["ihlal_turleri"].items()))

    if not o["gecerli_model_olcumu"]:
        print(f"\n  [!] GEÇERSİZ ÖLÇÜM: sağlayıcı {o['llm_hatasi_kosusu']} koşuda "
              "yanıt vermedi (kota/limit/ağ); yukarıdaki oranlar yedek yolu gösterir.")

    sizinti = o["izinsiz_musteriye_kampanya_sizintisi"]
    if sizinti:
        print(f"\n  izinsiz müşteriye kampanya sızıntısı: VAR -> {sizinti}")
    elif not o["izinsiz_musteriye_oneri_kosusu"]:
        print("\n  izinsiz müşteriye kampanya sızıntısı: ÖLÇÜLMEDİ "
              "(örnekte izinsiz müşteri yok)")
    else:
        print(f"\n  izinsiz müşteriye kampanya sızıntısı: YOK "
              f"({o['izinsiz_musteriye_oneri_kosusu']} öneri koşusunda)")

    if args.out:
        out_dir = resolve_path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        damga = sonuc["calisma_zamani"][:19].replace(":", "").replace("-", "")
        ham = out_dir / f"agent_eval_{damga}.json"
        ham.write_text(json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")
        etiket = (config["agent"]["groq"].get("model", "").split("/")[-1]
                  if config["agent"]["provider"] != "mock" else "mock")
        rapor = out_dir / f"agent_eval_{etiket}.md"
        rapor.write_text(markdown_rapor(sonuc), encoding="utf-8")
        print(f"\n  yazıldı: {ham}\n  yazıldı: {rapor}")

    return 1 if sizinti else 0


if __name__ == "__main__":
    sys.exit(main())

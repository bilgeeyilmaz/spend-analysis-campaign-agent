# Ajan Değerlendirme Raporu

> ⚠️ **Bu koşu geçerli bir model ölçümü değildir.** Sağlayıcı 16 koşuda yanıt veremedi (kota/limit/ağ) ve sistem deterministik yedeğe düştü. Aşağıdaki oranlar modelin değil, yedek yolun davranışını gösterir.

- Çalışma: `2026-08-20T11:03:38.250370+00:00`
- Sağlayıcı: **groq** · model `openai/gpt-oss-120b`
- Örnek: 4 müşteri × 4 istek = **16 koşu** (seed 42)

## Cevap durumu

| Durum | Adet | Oran |
|---|---:|---:|
| temiz | 0 | %0 |
| duzeltildi | 0 | %0 |
| yedek | 16 | %100 |
| hata | 0 | %0 |

**Müşteriye giden doğrulanmış metin oranı: %0** (denetimden geçen + düzelttirilip geçen). Kalanına şablon metin gitti — yani hiçbir koşuda doğrulanmamış rakam müşteriye ulaşmadı.

## Denetimin yakaladıkları

| İhlal | Adet |
|---|---:|
| `llm_hatasi` | 16 |

## Araç kullanımı

| Araç | Çağrı |
|---|---:|
| `analyze_spending` | 16 |
| `get_campaign_details` | 6 |
| `score_campaigns` | 4 |

## Süre ve tur

- Ortalama 0.24 sn · medyan 0.1 sn · en yavaş 2.32 sn
- Ortalama araç turu: 2.12

## Sert değişmez

✅ İzin vermeyen müşterinin cevabında kampanya kodu geçmedi (1 öneri koşusu, 4 koşu toplam).

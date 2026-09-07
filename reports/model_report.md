# Model Değerlendirme Raporu

## Kampanya kabul eğilimi modeli

- Algoritma: **lightgbm** · `{'learning_rate': 0.01, 'num_leaves': 4, 'n_estimators': 400}`
- Eğitim: 7,857 (müşteri, kampanya) çifti · 1,381 müşteri · 87 özellik
- Kabul oranı: %24.9 · eğitim tarihi 2026-08-19

| Metrik | Değer |
|---|---:|
| AUC | **0.778** |
| PR-AUC | 0.553 |
| precision@1 | 0.502 (239 müşteri) |
| Rastgeleye karşı kazanç | 1.85× |

### Baseline karşılaştırması

| Yöntem | AUC | PR-AUC | precision@1 |
|---|---:|---:|---:|
| rastgele | 0.502 | 0.263 | 0.272 |
| kampanya_populerligi | 0.654 | 0.364 | 0.368 |
| kategori_kurali | 0.769 | 0.567 | 0.498 |
| **model** | **0.778** | **0.553** | **0.502** |

Tek satırlık kategori kuralı PR-AUC'de modeli yeniyor (0.567 > 0.553). Model AUC ve precision@1'de önde. Bu satır raporda bilerek duruyor: baseline'ı gizlemek modeli olduğundan iyi gösterir.

### Sızıntı kontrolü

- Müşteri-gruplu bölme (raporlanan): **0.7782**
- Rastgele satır bölmesi: 0.7817 (893 müşteri hem eğitimde hem testte)
- Fark: **+0.0034** · gözlemlenebilir tavan 0.784

Bir müşterinin tüm teklifleri aynı müşteri bloğunu taşır; satır bazlı bölme modelin kişiyi tanıyıp üretecin sakladığı gizli eğilimi ezberlemesine izin verir. Fark sıfıra yakınsa model kişiyi değil deseni öğrenmiştir.

## Davranışsal segmentasyon

- Seçilen k: **3** · PCA 13 bileşen (açıklanan varyans %92.5)
- Davranışsal özellik: 22 · eğitim tarihi 2026-08-14

### k seçimi

| k | Silhouette | En küçük küme | Sonuç |
|---:|---:|---:|---|
| 3 | 0.194 | %19.2 | **seçildi** |
| 4 | 0.187 | %15.2 | — |
| 5 | 0.198 | %1.1 | elendi (< %5.0) |
| 6 | 0.168 | %1.1 | elendi (< %5.0) |
| 7 | 0.160 | %1.1 | elendi (< %5.0) |

En yüksek silhouette k=5'te (0.198) ama en küçük kümesi nüfusun %1.1'i — kampanya kurgulanamayacak bir artık. Segment bir hedef kitledir; iş kısıtı silhouette'ten önce gelir.

### Bulunan segmentler

| Küme | Ad | Müşteri |
|---:|---|---:|
| 0 | orta segment dengeli harcayan | 1,105 (55%) |
| 1 | yüksek harcamalı seyahat eden | 385 (19%) |
| 2 | orta segment dijital alışverişçi | 510 (26%) |


# Claim-Evidence Matrix

Bu dosya, mevcut `runs/` sonuçlarını ve `paper_package/validation_selected_summary_by_model.csv` içindeki validation-seçimli özetleri birlikte kullanarak hangi iddiaların güvenle yazılabileceğini özetler.

## Okuma kuralı

- `Kanıt`: doğrudan sonuç dosyalarından görülen bulgu
- `Yorum`: veriden çıkan ama neden-sonuç olarak ispatlanmamış çıkarım
- `Durum`: `güvenli`, `dikkatli yaz`, `kaçın`

## Matris

| İddia | Kanıt | Karşı örnek / nüans | Durum | Önerilen yazım |
| --- | --- | --- | --- | --- |
| DANN ailesi, parametre-eşleşmiş MLP'ye karşı çoğu koşulda daha iyi sonuç verebilir. | Validation-seçimli okumada Fashion full, KMNIST full, Fashion 0.2, CIFAR full ve CIFAR 0.2 koşularında en iyi ilgili DANN modeli `MLP_PARAM` üstündedir. | Fashion 0.1 koşusunda `DANN_LRF`, `MLP_PARAM` üstünde değildir. | `güvenli` | "Across five of the six evaluated benchmark conditions, a DANN variant outperformed the parameter-matched MLP baseline." |
| `DANN_LRF`, FashionMNIST full koşusunda en güçlü düşük-parametreli modeldir. | Validation-seçimli özet: `DANN_LRF = 0.87035`, `DANN_GRF = 0.86857`, `NAIVE_BRANCH = 0.86666`, `DANN_RANDOM = 0.86485`, `MLP_PARAM = 0.85362`. | `VANN_SAME` daha yüksek mutlak accuracy verir ama parametre sayısı çok daha yüksektir. | `güvenli` | "On FashionMNIST full, DANN_LRF was the strongest low-parameter model among the evaluated controls and DANN variants." |
| `DANN_LRF`, naive branching'in ötesinde ek fayda sağlayabilir. | Fashion full, Fashion 0.2, CIFAR full ve CIFAR 0.2 koşularında `DANN_LRF`, `NAIVE_BRANCH` üstündedir. | Farklar küçüktür. KMNIST full'de `DANN_LRF` ve `NAIVE_BRANCH` neredeyse eşittir. Fashion 0.1'de `DANN_LRF` geridedir. | `dikkatli yaz` | "DANN_LRF often, but not universally, improved over the naive branching control, suggesting that dendrite-level nonlinear integration can add value beyond branching alone." |
| En iyi dendritik örnekleme stratejisi veri yapısına bağlıdır. | Fashion full'de `DANN_LRF` önde; KMNIST full'de `DANN_RANDOM` önde. | Bu gözlem mekanizma ispatı değildir. | `güvenli` | "The best-performing dendritic sampling strategy was dataset-dependent in the present benchmark." |
| `DANN_RANDOM`, KMNIST için `DANN_LRF`'den daha uygundur. | Validation-seçimli özet: `DANN_RANDOM = 0.81427`, `DANN_LRF = 0.80913`. | Fark çok büyük değildir. | `güvenli` | "On KMNIST, the random sampling variant slightly outperformed DANN_LRF." |
| `DANN_LRF`, orta düzey low-data koşulunda faydasını korur. | Historical Fashion 0.2 ve CIFAR 0.2 koşularında `DANN_LRF` küçük farklarla öndedir. | Bu koşular train-only low-data değil, reduced-dataset diagnostic koşulardır. | `dikkatli yaz` | "In the current reduced-dataset diagnostic runs, DANN_LRF retained a small advantage in the 20% settings." |
| `DANN_LRF`, çok düşük veri koşulunda da en iyi modeldir. | Veri bunu desteklemiyor. Fashion 0.1'de validation-seçimli kontrolde lider `NAIVE_BRANCH`. | Bu negatif bulgu açıkça yazılmalıdır. | `kaçın` | Bu iddia yazılmamalıdır. |
| `DANN_LRF`, CIFAR-10'da parametre-eşleşmiş MLP'den açık biçimde daha iyidir. | Validation-seçimli özet: CIFAR full `0.47692` vs `0.30651`; CIFAR 0.2 `0.39665` vs `0.26485`. | Mutlak accuracy halen modern görsel modellerin gerisindedir. | `güvenli` | "DANN_LRF substantially outperformed the parameter-matched MLP on both evaluated CIFAR-10 settings." |
| `VANN_SAME`, daha yüksek kapasite ile daha yüksek mutlak accuracy sağlayabilir. | Fashion full, KMNIST full, CIFAR full ve CIFAR 0.2'de `VANN_SAME` en yüksek mutlak accuracy değerine sahiptir. | Bu model parametre-eşleşmiş değildir ve DANN katkısını geçersiz kılmaz. | `güvenli` | "The higher-capacity dense baseline achieved the highest absolute accuracy, but at a much larger parameter budget." |
| `DANN_LRF` hesaplama açısından bedelsizdir. | Timing bunu desteklemiyor. `DANN_LRF`, `MLP_PARAM`'dan daha yavaştır. | `NAIVE_BRANCH`'e göre ek maliyet sınırlıdır. | `kaçın` | Bu iddia yazılmamalıdır. |
| `DANN_LRF`, `NAIVE_BRANCH`'e göre sınırlı ek runtime maliyetiyle gelir. | Fashion ve CIFAR timing sonuçlarında train ve inference farkları küçüktür. | Timing ayrı bir CPU-only süreçte ölçülmüştür. | `güvenli` | "Relative to the naive branching control, DANN_LRF incurred only limited additional CPU runtime overhead." |
| Çalışma state-of-the-art görsel sınıflandırma sunmaktadır. | Veri bunu desteklemiyor. | CIFAR-10 absolute accuracy düşük, modeller flattened-input tabanlı. | `kaçın` | Bu iddia yazılmamalıdır. |

## Validation-seçimli sağlamlık kontrolü

`paper_package/validation_selected_summary_by_model.csv` dosyasında, `best_val_epoch` üzerindeki test accuracy özetleri çıkarıldı.

Bu ek kontrolün ana sonucu:

- Koşu liderleri değişmiyor.
- Ana full-data hikayesi korunuyor.
- Low-data bölümündeki temel yönler korunuyor, ancak bu bölüm train-only robustness kanıtı olarak değil diagnostic sonuç olarak sunulmalı.
- En dikkat çekici sayısal kayma `VANN_SAME` için, `max test` ile validation-seçimli skor arasındaki farkın daha belirgin olmasıdır.

## Kısa sonuç

Makale için güvenli ana omurga şudur:

1. Çalışma, DANN ailesini parameter-efficient benchmark çerçevesinde inceliyor.
2. `DANN_LRF` çoğu koşulda güçlü.
3. `DANN_RANDOM`, KMNIST gibi veri yapısının daha düzensiz olabileceği koşullarda öne çıkabiliyor.
4. `DANN_LRF` ile `NAIVE_BRANCH` arasındaki farklar küçük ama bilimsel olarak anlamlı olacak kadar yönlü.
5. Sonuçlar olumlu ama evrensel üstünlük iddiası desteklenmiyor.
6. Mevcut low-data kanıtı yardımcı nitelikte; güçlü low-training-data iddiası için fixed-test rerun tercih edilmelidir.

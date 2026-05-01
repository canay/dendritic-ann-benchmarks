# Rerun Notes

Bu dosya, mevcut benchmark paketindeki kod düzeltmelerinden sonra hangi koşuların yeniden çalıştırılmasının anlamlı olacağını özetler.

Bu not dosyası hazırlanırken hiçbir deney yeniden çalıştırılmadı.

## Neden yalnızca belirli koşular?

- Low-data akışında test seti artık varsayılan olarak sabit kalmaktadır.
- Ana rapor metriği artık `test_acc_at_best_val` olarak tasarlanmıştır.
- Seed, model kurulmadan önce de set edilmektedir.

Bu nedenle yeniden koşu alınacaksa öncelik düşük-veri koşularında olmalıdır.

## Önerilen öncelik

1. `FashionMNIST 0.2`
2. `FashionMNIST 0.1`
3. `CIFAR-10 0.2`

## Önerilen çıktı klasörleri

- `runs_fashion_low02_fixedtest`
- `runs_fashion_low01_fixedtest`
- `runs_cifar_low02_fixedtest`

## Örnek komutlar

PowerShell içinde, `dann_benchmark` klasöründe:

```text
python benchmark.py --dataset fashionmnist --data-root ./data --output-dir ./runs/runs_fashion_low02_fixedtest --subset-fraction 0.2 --epochs 30 --batch-size 256 --seeds 0 1 2 3 4 5 6 7 8 9 --device cpu --report-metric test_acc_at_best_val
```

```text
python benchmark.py --dataset fashionmnist --data-root ./data --output-dir ./runs/runs_fashion_low01_fixedtest --subset-fraction 0.1 --epochs 30 --batch-size 256 --seeds 0 1 2 3 4 5 6 7 8 9 --device cpu --report-metric test_acc_at_best_val
```

```text
python benchmark.py --dataset cifar10 --data-root ./data --output-dir ./runs/runs_cifar_low02_fixedtest --subset-fraction 0.2 --epochs 30 --batch-size 256 --seeds 0 1 2 3 4 5 6 7 8 9 --device cpu --report-metric test_acc_at_best_val
```

## Legacy reproduction

Eski low-data davranışını bilinçli olarak yeniden üretmek istersen, `--test-subset-fraction` bayrağını ayrıca ver.

Örnek:

```text
python benchmark.py --dataset fashionmnist --output-dir ./runs/legacy_repro --subset-fraction 0.1 --test-subset-fraction 0.2 --report-metric best_test_acc
```

## Yorum

Mevcut eski low-data klasörleri tamamen atılmak zorunda değildir. Ancak güçlü bir "low-training-data robustness" iddiası kurulacaksa, ana kanıt olarak fixed-test yeniden koşuları tercih edilmelidir.

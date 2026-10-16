# MLE-AI-201 Final Project - Real Estate Price Prediction

Bu repozitoriya Bina.az elanları əsasında mənzillərin qiymətinin proqnozlaşdırılması və kateqoriyalaşdırılması (regressiya və təsnifat) layihəsi üçün yaradılmışdır.

## Layihə Strukturu
- `src/data_prep.py`: Xam datanın yüklənməsi, təmizlənməsi, feature engineering, train/val/test split və median imputasiya funksiyaları.
- `src/evaluate.py`: Reqressiya (RMSE, MAE, R2) və təsnifat (Precision, Recall, F1, ROC-AUC) metrikləri, həmçinin `compare()` cədvəl funksiyası.
- `src/run_all.py`: Bütün boru kəmərini (pipeline) avtomatik işə salan əsas skрыпт.
- `tests/`: Metriklərin düzgünlüyünü yoxlayan unit testlər.

## Quraşdırma və İşə Salınma

1. Virtual mühit yaradın və aktivləşdirin (Python 3.11 və ya 3.12 tövsiyə olunur):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows üçün: venv\Scripts\activate
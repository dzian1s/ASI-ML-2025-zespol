# Model Card

## 1. Problem & Intended Use

Model służy do **prognozowania ceny wynajmu mieszkań Airbnb w Nowym Jorku**.  
Celem jest wsparcie gospodarzy (hostów Airbnb) w ustaleniu optymalnej ceny ofertowej
na podstawie cech mieszkania, jego lokalizacji oraz popularności.

**Intended users:**
- gospodarze Airbnb
- analitycy rynku najmu
- systemy automatycznej rekomendacji cen

## 2. Data (source, license, size, PII=no)

**Source:**  
New York City Airbnb Open Data  
(https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)

**License:**  
CC0: Public Domain  
https://creativecommons.org/publicdomain/zero/1.0/

**Dataset size:**  
- ~49 000 ofert  
- 16 kolumn  
- po czyszczeniu: ~35 000 ofert

**PII:**  
Dane nie zawierają danych osobowych (wszystkie identyfikatory hostów zostały usunięte podczas przetwarzania).

**Features użyte w modelu:**  
- neighbourhood_group  
- neighbourhood  
- room_type  
- minimum_nights  
- number_of_reviews  
- reviews_per_month  
- calculated_host_listings_count  
- availability_365  
- (target) price

## 3. Metrics (main + secondary)

**Train/test split:**
- 80/20
- random_state = 42
- podział losowy (stratyfikacja nie była wymagana, bo target = price jest ciągły)

**Główna metryka:**  
- **RMSE** (Root Mean Squared Error)

**Wynik modelu produkcyjnego:**  
- **RMSE (test split): 60.21**  
(otrzymany przez najlepszy run AutoGluon)

**Metryki pomocnicze:**  
- czas trenowania: ~56 s  
- czas predykcji modelu: ~3 ms / próbkę  
- stabilność: różnica RMSE pomiędzy runami ~2-3 punkta

## 4. Limitations & Risks

### Ograniczenia:
- model trenowany wyłącznie na **danych z Nowego Jorku** – nie uogólnia się na inne miasta
- brak informacji o dacie rezerwacji → model **nie uwzględnia sezonowości** cen
- dane historyczne mogą być nieaktualne  
- model nie bierze pod uwagę:
  - jakości zdjęć
  - opinii tekstowych
  - zmian w trendach rynkowych
  - wydarzeń specjalnych (święta, maratony, konferencje)

### Ryzyka:
- model może **zawyżyć lub zaniżyć rekomendowaną cenę**
- ryzyko błędnej interpretacji przez użytkownika
- możliwość błędu dla nietypowych mieszkań (outliers)

### Jak zmniejszyć ryzyko:
- stosować sanity-check (minimum i maksimum ceny)
- ograniczyć zakres predykcji np. 10 USD → 500 USD
- prezentować wynik jako *przedział cen*, a nie jedną liczbę
- okresowa rewalidacja modelu (co 3–6 miesięcy)

## 5. Ethics & Risk Considerations

- model może wzmacniać podział na “droższe” i “tańsze” dzielnice → **potencjalny bias lokalizacyjny**
- istnieje ryzyko, że rekomendacje modelu mogą wpływać na ceny najmu w niektórych obszarach (niezamierzone skutki społeczne)
- dane pochodzą z rynku z nierównym rozkładem (manhattan ma przewagę)

Mitigacje:
- monitorowanie metryk stabilności i błędu per-neighbourhood
- stosowanie regularnych aktualizacji danych
- prezentowanie wyników jako narzędzie pomocnicze, nie decydujące

## 6. Versioning (model + data + code)

**W&B run (model produkcyjny):**  
https://wandb.ai/s27335-polsko-japo-ska-akademia-technik-komputerowych/asi-ml-2025-zespol/runs/p4py0hol?nw=nwusers25282

**Model artifact (production):**  
`ag_model:production`  
(alias: `production`)

**Data artifact:**  
`clean_data:v1`  
(wersja danych użytych do treningu)

**Git commit:**  
`Commit 5ddbe24`
SHA: 5ddbe24b322b970e5ae1c8dbdae0ed786cf3192b

**Środowisko:**  
- Python 3.11  
- AutoGluon 1.x  
- scikit-learn 1.5  
- Kedro 1.0  
- numpy, pandas  

# Design proposal

Projekt ma za zadanie przeprowadzić ewaluację dostępnych rozwiązań detekcji sztucznie wygenerowanej muzyki. Badania będą prowadzone pod kątem wyjaśnialności otrzymywanych predykcji w celu analizy jakości obecnych metod na podstawie dotychczasowych przemyśleń oraz wniosków [[1](#ref-1)] [[2](#ref-2)]. W ramach niniejszej pracy wybrano szereg podejść - modele Random Forest, SVM i kNN [[3](#ref-3)]; model typu Transformer _SONICS_ [[5](#ref-5)] oraz sieć konwolucyjna zespołu _Deezer_ [[4](#ref-4)]. Wszystkie z wymienionych opcji zostały zastosowane w celu klasyfikacji piosenek wygenerowanych przez ogólnodostępne platformy generujące muzykę, takie jak _Suno_, _Udio_ czy też _Riffusion_. Kierując się podejściem z pracy [[6](#ref-6)] przeprowadzone zostaną eksperymenty na dostępnych modelach.

W ramach rozszerzenia projektu wykonane zostanie również porównanie, czy podstawowe modele klasyfikacji [[3](#ref-3)] są funkcjonalnie podobne do IRCAM Amplify.

## Wykorzystane technologie / Stack technologiczny

- Podstawowy stack machine learningowy: python, numpy, matplotlib/seaborn lub plotly
- Dla keżdego z modeli wykorzystamy rózne techniki wyjaśnialności:
- RF, SVM i kNN [[3](#ref-3)] - SHAP, LIME
- CNN [[4](#ref-4)] - Grad-cam
- SONICS (transformer) [[5](#ref-5)] - analiza map samoatencji (self-attention)

## Plan pracy

1. **03.11 - 09.11**

   - Przeczytanie literatury, odnalezienie istniejących rozwiązań
   - Zapoznianie się z narzędziami do wyjaśnialności, które można zastosować
   - Design proposal - deadline 05.11, feedback 07.11 (piątek)

2. **10.11 - 16.11**

   - Głębsza analiza literatury i napisanie szerszych wniosków
   - Przygotowanie trzech wstępnych eksperymentów na modelach dostępnych w repozytorium <https://github.com/lcrosvila/ai-music-detection>
   - Prezentacja prototypu (deadline do 14.11)

3. **17.11 - 23.11**

   - Rozszerzenie eksperymentów na pozostałe modele

4. **24.11 - 30.11**

   - Kontynuacja

5. **01.12 - 07.12**

   - Zintegrowanie zestawów danych oraz ekstrakcja wniosków z uzyskanych opisów wyjaśnialności modeli pod kątem uchwycenia elementów wspólnych dla piosenek wygenerowanych za pomocą modeli generatywnych

6. **08.12 - 14.12**

   - Kontynuacja

7. **15.12 - 21.12**

   - Podsumowanie wniosków, wizualizacja otrzymanych rozwiązań pod kątem wyjaśnialności uzyskanego wyniku względem poszczególnych generatorów piosenek

8. **22.12 - 28.12**

   - Święta, przerwa lub nadgonienie opóźnień

9. **29.12 - 4.01**

   - Przygotowanie artykułu naukowego

10. **05.01 - 11.01**

    - Kontynuacja

11. **12.01 - 18.01**

    - Przygotowanie filmiku
    - Deadline złożenia projektu 15.01 (czwartek)

12. **19.01 - 25.01**

    - Przygotowanie prezentacji

13. **26.01 - 01.02**

    - Przygotowanie prezentacji
    - Prezentacja 28.01

## Bibliografia

<a id="ref-1"></a>**[1]** Afchar, Darius, et al. A Fourier Explanation of AI-music Artifacts. ISMIR 2025 (Best Paper). <https://arxiv.org/abs/2506.19108>

<a id="ref-2"></a>**[2]** Sroka, Tomasz, et al. Evaluating Fake Music Detection Performance Under Audio Augmentations. ISMIR 2025 Late-Breaking Demo. <https://arxiv.org/pdf/2507.10447>

<a id="ref-3"></a>**[3]** Cros Vila, Laura, et al. (2025). The AI Music Arms Race: On the Detection of AI-Generated Music. Transactions of the International Society for Music Information Retrieval (TISMIR) 8(1). <https://transactions.ismir.net/8/volume/8/issue/1>

<a id="ref-4"></a>**[4]** Afchar, Darius; Meseguer-Brocal, Gabriel; Hennequin, Romain (2024). Detecting Music Deepfakes is Easy but Actually Hard. arXiv. <https://arxiv.org/abs/2405.04181>

<a id="ref-5"></a>**[5]** Rahman, M. A., Hakim, Z. I. A., Sarker, N. H., Paul, B., & Fattah, S. A. (2024). SONICS: Synthetic Or Not--Identifying Counterfeit Songs. arXiv preprint arXiv:2408.14080.

<a id="ref-6"></a>**[6]** Li, Y., Sun, Q., Li, H., Specia, L., & Schuller, B. W. (2024). Detecting Machine-Generated Music with Explainability--A Challenge and Early Benchmarks. arXiv preprint arXiv:2412.13421.<https://arxiv.org/abs/2412.13421>

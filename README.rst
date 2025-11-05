TODO:
  * dokładny podział na tygodnie
  * podział pracy

===============
Design proposal
===============
Projekt ma za zadanie przeprowadzić ewaluację dostępnych rozwiązań detekcji sztucznie wygenerowanej muzyki. Badania będą prowadzone pod kątem wyjaśnialności otrzymywanych predykcji w celu analizy jakości obecnych metod na podstawie dotychczasowych przemyśleń oraz wniosków [1]_[2]_. W ramach powyższej pracy wybrano szereg podejść - modele Random Forest, SVM i kNN [3]_; model typu Transformer *SONICS* [5]_ oraz sieć konwolucyjna zespołu *Deezer* [4]_. Wszystkie z wymienionych opcji zostały zastosowane w celu klasyfikacji piosenek wygenerowanych przez ogólnodostępne platformy generujące muzykę, takie jak *Suno*, *Udio* czy też *Riffusion*.

====================
Wykorzystane technologie / Stack technologiczny
====================
* Tutaj może wskazać narzędzia do wyjaśnialności, których użyjemy
* Stack będzie taki, jakie mają te narzędzia
* pajtong+matplotlib/plotly do wykresów (to może być dosyć ważny punkt projektu, bo w sumie na tym polega wyjaśnialność żeby to ładnie pokazać) 

=======================
Plan pracy
=======================
#. 29.09 - 05.10
  * X
#. 06.10 - 12.10
  * X
#. 13.10 - 19.10
  * X
#. 20.10 - 26.10
  * X
#. 27.10 - 02.11
  * X
#. 03.11 - 09.11
  * Przeczytanie literatury, odnalezienie istniejących rozwiązań
  * Zapoznianie się z narzędziami do wyjaśnialności, które można zastosować
  * Design proposal - deadline 05.11, feedback 07.11 (piątek)
#. 10.11 - 16.11
  * Głębsza analiza literatury i napisanie szerszych wniosków
  * "funkcjonalny prototyp projektu, postęp analizy literaturowej, konfigurację środowiska eksperymentalnego."
  * Prototyp - deadline na spotkania 14.11 (piątek) ??
#. 17.11 - 23.11
  * H
#. 24.11 - 30.11
  * I
#. 01.12 - 07.12
  * J
#. 08.12 - 14.12
  * K
#. 15.12 - 21.12
  * L
#. 22.12 - 28.12
  * Święta, przerwa lub nadgonienie opóźnień
#. 29.12 - 4.01
  * N
#. 05.01 - 11.01
  * O
#. 12.01 - 18.01
  * Przygotowanie filmiku
  * >>>> Deadline zwolnienia z kolosa 15.01 (czwartek)
#. 19.01 - 25.01
  * XXXX Deadline Finalny (piątek)
  * przygotowanie prezentacji
#. 26.01 - 01.02
  * Przygotowanie prezentacji
  * Prezentacja 28.01

Bibliografia
.. [1] Afchar, Darius, et al. A Fourier Explanation of AI-music Artifacts. ISMIR 2025 (Best Paper). https://arxiv.org/abs/2506.19108
.. [2] Sroka, Tomasz, et al. Evaluating Fake Music Detection Performance Under Audio Augmentations. ISMIR 2025 Late-Breaking Demo. https://arxiv.org/pdf/2507.10447
.. [3] Cros Vila, Laura, et al. (2025). The AI Music Arms Race: On the Detection of AI-Generated Music. Transactions of the International Society for Music Information Retrieval (TISMIR) 8(1). https://transactions.ismir.net/8/volume/8/issue/1
.. [4] Afchar, Darius; Meseguer-Brocal, Gabriel; Hennequin, Romain (2024). Detecting Music Deepfakes is Easy but Actually Hard. arXiv. https://arxiv.org/abs/2405.04181
.. [5] Rahman, M. A., Hakim, Z. I. A., Sarker, N. H., Paul, B., & Fattah, S. A. (2024). SONICS: Synthetic Or Not--Identifying Counterfeit Songs. arXiv preprint arXiv:2408.14080.

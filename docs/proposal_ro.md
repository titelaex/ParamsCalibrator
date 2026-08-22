# ParamsCalibrator — Propunere Tehnică

**Siemens Curious Minds Software Summer School 2026 — Categorie: Procesare Semnal & Algoritmi**
**Autor:** Alexandra | **Termen:** 28 August 2026, ora 13:00

---

## 1. Justificarea categoriei

Proiectul a fost încadrat inițial la categoria Digital Twins & Platforme, dar la o analiză mai atentă, potrivirea este slabă: descrierea acelei categorii se concentrează pe crearea replicii virtuale în sine — o simulare care rulează în timp real a unui sistem fizic, cu accelerare pe GPU și fluxuri mari de date. Acest proiect nu construiește o replică funcțională; el calibrează constantele fizice care alimentează una. Potrivirea literală este cu Procesare Semnal & Algoritmi: "achiziția și procesarea datelor, optimizare matematică și dezvoltare de modele simplificate pentru fenomene complexe" — o descriere aproape exactă a ceea ce face acest proiect (date brute de la senzori la intrare, optimizare matematică/calibrare, un model ML ca model simplificat surogat). Repoziționarea sub această categorie elimină nevoia de a forța încadrarea și permite evaluarea proiectului pentru ceea ce este cu adevărat.

## 2. Descrierea problemei

Calibrarea constantelor fizice (rate de transfer termic, rezistențe termice, coeficienți de amortizare, coeficienți de frecare etc.) pe baza datelor de la senzori este o problemă inginerească recurentă: un model este la fel de bun ca parametrii introduși în el, iar acești parametri variază în timp din cauza uzurii, variațiilor de fabricație și condițiilor de operare în schimbare. În prezent, această calibrare se bazează de obicei pe optimizare iterativă — algoritmi genetici, metode de tip least-squares bazate pe gradient, filtre secvențiale sau alți solveri euristici/numerici — fiecare rulat de la zero, cu un cost computațional real, de fiecare dată când trebuie procesate date noi.

**Oportunitate:** înlocuirea căutării iterative repetate cu un model ML antrenat care mapează direct o fereastră de date brute de la senzori la constantele calibrate, printr-o singură trecere înainte (forward pass), și evaluarea riguroasă a acestuia în raport cu familiile clasice de metode de calibrare — nu doar algoritmi genetici, ci întregul spectru folosit efectiv în practica inginerească.

## 3. Soluția propusă

Un model de Machine Learning antrenat pentru a realiza calibrarea parametrilor, evaluat comparativ cu cinci metode clasice de calibrare pe mai multe axe, expus printr-un API rapid:

- **Intrare:** o fereastră de date brute/zgomotoase de la senzori (de ex. temperatura în timp, sub un profil de sarcină cunoscut), plus date de excitație cunoscute (sarcina aplicată, condițiile ambientale).
- **Ieșire:** constanta (constantele) fizică calibrată (de ex. coeficientul de transfer termic, rezistența termică).
- **Ipoteza de validat:** abordarea ML egalează sau depășește metodele clasice de calibrare în privința acurateței, fiind în același timp semnificativ mai rapidă — pentru că face munca costisitoare o singură dată, offline, în timpul antrenării, iar fiecare calibrare ulterioară este doar o trecere înainte ieftină. Aceasta este comparația centrală pe care proiectul trebuie să o demonstreze, nu doar să o afirme.

## 4. Banc de testare: Modelul termic al înfășurării unui motor

Nu există date sau echipamente proprietare Siemens disponibile pentru acest proiect, așa că bancul de testare folosește **date sintetice generate dintr-un model fizic realist**, cu parametri extrași din intervale reprezentative pentru literatura de specialitate privind proiectarea motoarelor (de ex. clasele termice și constantele de timp menționate în IEC 60034), nu din cifre confidențiale — menționat explicit ca ipoteză de modelare.

**Sistem:** o înfășurare de motor electric care se încălzește sub sarcină, răcită prin convecție — o instanță concretă, bine documentată, a problemei generale de calibrare, relevantă pentru divizia de motoare și acționări a Siemens.

**Model cu 1 parametru (concentrat, un singur nod):**

```
C · dT/dt = I(t)² · R_înfășurare − h·A · (T − T_ambient)
```

Ținta calibrării: `h·A` (coeficientul de transfer termic concentrat).

**Model cu mai mulți parametri (2 noduri: înfășurare + carcasă):** adaugă o a doua masă termică și o cale de conducție înfășurare→carcasă, rezultând 2–3 constante de calibrat simultan — această variantă alimentează comparația de *scalabilitate* (Secțiunea 6).

**Generarea datelor:** se simulează mai multe profiluri de sarcină (sarcini în treaptă, cicluri de funcționare, rampe), se eșantionează constante reale (ground-truth) din intervale realiste, se adaugă zgomot gaussian la mai multe niveluri, iar un set de testare este păstrat separat, la care modelul ML nu are acces în timpul antrenării.

## 5. Metode de calibrare de referință (baseline)

Cinci metode, alese astfel încât să acopere familiile distincte folosite efectiv în calibrarea inginerească, astfel încât comparația să acopere întregul domeniu, nu doar un singur baseline convenabil:

| Metodă | Familie | De ce este inclusă |
|---|---|---|
| Algoritm Genetic (GA) | Euristică bazată pe populație | Baseline-ul inițial; abordare evolutivă standard |
| Particle Swarm Optimization (PSO) | Euristică bazată pe populație | O a doua metaeuristică, distinctă — arată că modelul ML depășește întreaga familie de euristici, nu doar un membru al ei |
| Levenberg-Marquardt (LM) | Regresie neliniară clasică (least-squares bazat pe gradient) | Alegerea implicită a majorității inginerilor pentru calibrare prin curve-fitting; potrivește forma exactă a ecuației fizice cunoscute pe fiecare curbă nouă, o calibrare la un moment dat — cel mai apropiat "văr" conceptual al modelului ML (Secțiunea 6) |
| Extended Kalman Filter (EKF) | Estimare secvențială/online | Metoda clasică proiectată efectiv pentru actualizarea parametrilor în timp real, în flux continuu — cel mai direct concurent al modelului ML în privința vitezei și a capacității de streaming |
| Bayesian Optimization | Metodă modernă, eficientă din punct de vedere al eșantioanelor, fără gradient | Reprezintă practica modernă pentru probleme de calibrare unde fiecare evaluare este costisitoare |

## 6. Modelul ML: Design

**Arhitectură: un MLP mic peste caracteristici (features) motivate fizic**, nu o rețea profundă pe semnalul brut. Fizica din spatele acestei probleme este un răspuns termic simplu și predictibil (o creștere către un platou), deci este puțin de câștigat dintr-o arhitectură mai grea (de ex. CNN) construită pentru a găsi tipare în semnale brute complexe sau de dimensiune mare — un set compact de caracteristici este suficient și mult mai ușor de antrenat bine în timpul disponibil.

**Caracteristici de intrare**, extrase pentru fiecare fereastră de senzor:

- panta inițială a creșterii temperaturii
- temperatura de regim staționar estimată (când sarcina este menținută suficient de mult timp)
- constanta de timp termică estimată (timpul necesar pentru a atinge ~63% din variația totală de temperatură — caracterizarea standard a unui sistem de ordinul întâi)
- curentul aplicat și temperatura ambientală pentru acea rulare
- un indicator al nivelului de zgomot (varianța locală), astfel încât modelul să poată estima fiabilitatea datelor

Acestea nu sunt alese arbitrar: fizica oferă deja o relație verificabilă la regim staționar, `T_ss − T_ambient = I²R / (h·A)`, deci `h·A` poate fi, în principiu, calculat direct algebric atunci când sarcina este menținută constantă suficient de mult timp. Rolul MLP-ului este să învețe versiunea mai generală și mai robustă a acestei relații — una care rămâne validă pe date zgomotoase, ferestre scurte/incomplete și condiții care nu sunt de regim staționar, unde formula închisă nu se aplică direct. Această încadrare face modelul și mai ușor de explicat în prezentare: nu este o cutie neagră, ci o generalizare învățată a unei formule fizice cunoscute.

**Rețea:** caracteristici (~8–12 intrări) → Dense(32, ReLU) → Dense(16, ReLU) → strat de ieșire cu câte o unitate pentru fiecare constantă calibrată (1 pentru bancul de testare cu un singur nod, 2–3 pentru cel cu mai multe noduri). Antrenat cu funcția de cost MSE, normalizată per constantă (deoarece constantele diferite au scale numerice foarte diferite), folosind Adam. Suficient de mică pentru a fi antrenată în câteva secunde până la câteva minute pe CPU.

**Notă conceptuală pentru prezentare:** acest model și LM (Secțiunea 5) sunt ambele, din punct de vedere tehnic, "regresie neliniară" — LM potrivește forma exactă a ecuației fizice cunoscute pe fiecare curbă nouă, de la zero, de fiecare dată; MLP-ul învață o aproximare flexibilă, de uz general, o singură dată, din mii de exemple simulate, și apoi doar evaluează. Aceeași categorie matematică, strategii diferite — un mod clar de a încadra comparația pentru un public non-tehnic.

**Alternativă de rezervă:** un MLP peste fereastra brută de serie temporală (fără extragere de caracteristici) este o alternativă mai simplă, dar care necesită mai multe date și e mai greu de explicat — păstrată doar ca Plan B.

**Obiectiv suplimentar (opțional, nu esențial):** Gaussian Process Regression ca model alternativ de regresie neliniară, care oferă gratuit un interval de încredere pentru fiecare constantă calibrată — util dacă rămâne timp pentru a explora calibrarea cu estimarea incertitudinii, dar nu este necesar pentru pitch-ul central.

## 7. Metrici de evaluare

Toate cele șase metode (5 baseline + modelul ML) sunt evaluate pe aceleași cazuri de test sintetice, pe următoarele axe:

1. **Acuratețe** — eroarea de calibrare față de constanta (constantele) reală(e).
2. **Viteză / latență** — timpul de calcul necesar pentru a produce o valoare calibrată.
3. **Robustețea convergenței** — rata de succes și varianța pe diferite estimări inițiale aleatorii și niveluri de zgomot (metodele euristice și cele bazate pe gradient pot să nu convergă sau să rămână blocate; acest lucru este expus onest).
4. **Scalabilitate** — cum se modifică fiecare metrică la trecerea de la bancul de testare cu 1 parametru la cel cu 2–3 parametri.
5. **Adecvarea pentru timp real/streaming** — dacă metoda poate fi actualizată incremental pe măsură ce sosesc date noi de la senzori, sau necesită o rerulare completă pe tot lotul de date. Este de așteptat ca EKF să fie cu adevărat competitiv aici, ceea ce întărește credibilitatea rezultatelor — un rezultat în care "modelul ML câștigă la toate capitolele" ar fi de fapt mai puțin convingător.

## 8. Arhitectura sistemului

```
┌──────────────────────┐
│  Simulator Fizic       │  model termic motor (1 nod & 2 noduri)
│  + Generator Date       │  → profiluri de sarcină, date senzor cu
│    Sintetice            │     zgomot, valori reale
└──────────┬─────────────┘
           │
           ▼
┌───────────────────────────────────────────────┐
│              Cadru de Benchmarking               │
│  rulează GA / PSO / LM / EKF / BayesOpt / MLP    │
│  pe cazuri de test identice, înregistrează        │
│  toate cele 5 metrici                             │
└──────────┬────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐      ┌───────────────────────────┐
│  Rezultate & Grafice   │      │  Extragere Caracteristici    │
│  (pentru prezentare)   │      │  → MLP Antrenat               │
└──────────────────────┘      │  → Wrapper FastAPI            │
                               │  → Microserviciu Dockerizat   │
                               │  (demo live: POST date senzor │
                               │   → valoare calibrată)        │
                               └───────────────────────────┘
```

## 9. Stack tehnologic

- **Simulare & baseline-uri:** Python, NumPy/SciPy (`scipy.optimize.least_squares` pentru LM), o implementare proprie ușoară sau bazată pe librărie pentru GA/PSO, `filterpy` pentru EKF, `scikit-optimize` sau `bayes_opt` pentru Bayesian Optimization.
- **Model ML:** MLP mic peste caracteristici extrase (scikit-learn `MLPRegressor` sau un model PyTorch minimal — oricare este suficient de ușor aici; scikit-learn este alegerea mai simplă având în vedere dimensiunea mică a rețelei).
- **Microserviciu:** FastAPI + Docker.
- **Vizualizare:** Python (matplotlib/plotly) pentru grafice de benchmarking; slide-urile prezentării finale.

## 10. Plan de etape (12 zile până la 28 august, ora 13:00)

| Zile | Etapă |
|---|---|
| 1–2 | Simulator fizic + generator de date sintetice funcțional de la un capăt la altul; validat prin verificări de sanitate |
| 3–4 | Metode baseline implementate (GA, PSO, LM) și funcționale pe date sintetice |
| 5 | Adăugarea baseline-urilor EKF și Bayesian Optimization |
| 6–7 | Pipeline de extragere a caracteristicilor + MLP antrenat și validat pe date de test separate |
| 8 | Rulare completă a cadrului de benchmarking pe toate cele 6 metode × 5 metrici × ambele variante de banc de testare |
| 9 | Microserviciu FastAPI dockerizat care încapsulează MLP-ul; demo live funcțional |
| 10 | Finalizarea graficelor cu rezultate; verificarea sanității cifrelor față de așteptări |
| 11 | Construirea prezentării (10 minute, cu accent pe problemă + soluție, conform cerințelor temei) |
| 12 | Rezervă / repetiție |

## 11. Livrabile

1. Repository de cod funcțional (`ParamsCalibrator`): simulator, baseline-uri, model ML, cadru de benchmarking.
2. Microserviciu dockerizat cu un demo API funcțional.
3. Rezultate de benchmarking și grafice comparative.
4. Prezentare de 10 minute.

## 12. Riscuri & Atenuare

- **Domeniu prea mare pentru 12 zile** → construire în ordinea etapelor de mai sus; un banc de testare cu 1 nod, cu toate cele 6 metode comparate pe acuratețe + viteză, este rezultatul minim viabil dacă timpul devine insuficient — axele de robustețe/scalabilitate/streaming pot fi eliminate primele, la nevoie.
- **Rezultatul "ML câștigă la toate capitolele" pare neconvingător** → păstrarea și prezentarea deliberată a rezultatului EKF pentru streaming chiar dacă este competitiv cu MLP-ul; un rezultat nuanțat este mai credibil decât o victorie totală.
- **Lipsa datelor reale Siemens** → menționată explicit ca ipoteză de modelare; bancul de testare este ancorat în intervale de parametri realiste, documentate public, nu inventate.
- **Schimbarea categoriei față de ce a fost validat cu Siemens** → merită o scurtă informare către persoana care a validat direcția Digital Twins, deoarece munca tehnică de bază rămâne neschimbată, doar încadrarea s-a mutat către o categorie care i se potrivește mai bine, literal.

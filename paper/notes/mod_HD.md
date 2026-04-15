# Модифицированная кривая HD

## 1 Вывод классической кривой

### 1.1 Система координат

$$
\hat{\Omega} = \left(\sin \theta \cos \phi, \sin \theta \sin \phi, \cos \theta \right)
$$

$$
\hat{m} = (\sin \phi, - \cos \phi, 0),
$$

$$
\hat{n} = (\cos \theta \cos \phi, \cos \theta \sin \phi, - \sin \theta),
$$

### 1.2 Поляризации

$$
e^+_{ij}(\Omega) = \hat{m}_i \hat{m}_j - \hat{n}_i \hat{n}_j,
$$

$$
e^{\times}_{ij}(\Omega) = \hat{m}_i \hat{n}_j - \hat{n}_i \hat{m}_j,
$$

### 1.3 Отклик пульсара

$$
F_{\hat{p}}^A(\Omega) = e^A_{ij}(\Omega) \frac{1}{2} \frac{\hat{p}^i \hat{p}^j}{1 + \hat{\Omega} \cdot \hat{p}}
$$

$$
F_{\hat{p}}(\Omega) = F_{\hat{p}}^+(\Omega) + i F_{\hat{p}}^{\times}(\Omega)
$$

$$
F_{\hat{p}}(\Omega) =  \frac{1}{2} \frac{(\hat{p} \cdot \hat{k})^2}{1 + \hat{\Omega} \cdot \hat{p}}
$$

$$
\hat{k} = \hat{m} + i \hat{n}
$$

### 1.4 Симметрии

При повороте ситемы координат вокруг вектора $\hat{\Omega}$ на угол $\psi$: $\hat{k} \rightarrow e^{i\psi} \hat{k}$, функция $F_{\hat{p}}(\Omega)$ преобразуется как:

$$
F_{\hat{p}}(\Omega) \rightarrow e^{i2\psi} F_{\hat{p}}(\Omega)
$$

Это называется spin-2 симметрия, что дает нам возможность представить функцию $F_{\hat{p}}(\Omega)$ в виде:

$$
F_{\hat{p}}(\Omega) = \sum_{l=2}^{\infty} \sum_{m = -l}^l a_{lm} \cdot {}_{2}Y_{lm}(\Omega)
$$

где ${}_{2}Y_{lm}(\Omega)$ - сферическая spin-2 гармоника. А поскольку от вектора $\hat{p}$ в функции зависимость только от скалярного произведения (угла между векторами), то коэффициенты $a_{lm}$ тоже можно представить в виде разложения на сферические функции. Итого:

$$
F_{\hat{p}}(\Omega) = \sum_{l=2}^{\infty} \sum_{m = -l}^l A_l \cdot {}_{2} Y_{lm}(\Omega) Y^*_{lm}(p)
$$

где:

$$
A_l = \frac{4 \pi (-1)^l}{\sqrt{(l+2)(l+1)l(l-1)}}
$$

### 1.5 Интегральный оператор

Расмотрим теперь интеграл для идеального PTA:

$$
\Gamma(p,q) = \frac{1}{4 \pi}\int_{S^2} d\Omega \cdot \mathcal{R} \left[ F_{\hat{p}}(\Omega) \cdot F^*_{\hat{q}}(\Omega) \right] \mathcal{P}(\Omega)
$$

Это можно переписать следующим образом:

$$
\Gamma(p,q) = \frac{1}{4 \pi} \mathcal{R} \left[  \int_{S^2} d\Omega F_{\hat{p}}(\Omega) \cdot F^*_{\hat{q}}(\Omega) \cdot \mathcal{P}(\Omega) \right]
$$

где:

$$
F_{\hat{p}}(\Omega) \cdot F^*_{\hat{q}}(\Omega) = \left[ \sum_{l=2}^{\infty} \sum_{m = -l}^l A_l \cdot {}_{2}Y_{lm}(\Omega) Y^*_{lm}(p) \right] \cdot \left[ \sum_{l'=2}^{\infty} \sum_{m' = -l'}^{l'} A_{l'} \cdot {}_{2}Y^{*}_{l'm'}(\Omega) Y_{l'm'}(q) \right]
$$

$$
F_{\hat{p}}(\Omega) \cdot F^*_{\hat{q}}(\Omega) = \sum_{l=2}^{\infty} \sum_{l'=2}^{\infty} \sum_{m = -l}^l  \sum_{m' = -l'}^{l'} A_l A_{l'} \cdot Y^*_{lm}(p) Y_{l'm'}(q) \cdot {}_{2}Y_{lm}(\Omega) {}_{2}Y^{*}_{l'm'}(\Omega)
$$

### 1.6 Классическа кривая

$$
\mathcal{P}(\Omega) = 1
$$

Тогда:

$$
\Gamma(p,q) = \frac{1}{4 \pi} \mathcal{R} \left[  \int_{S^2} d\Omega \cdot \sum_{l,l'} \sum_{m, m'} A_l A_{l'} \cdot Y^*_{lm}(p) Y_{l'm'}(q) \cdot  {}_{2}Y_{lm}(\Omega) {}_{2}Y^{*}_{l'm'}(\Omega) \right]
$$

$$
\Gamma(p,q) =  \frac{1}{4 \pi} \mathcal{R} \left[  \sum_{l,l'} \sum_{m, m'} A_l A_{l'} \cdot Y^*_{lm}(p) Y_{l'm'}(q) \cdot  \int_{S^2} d\Omega \cdot {}_{2}Y_{lm}(\Omega) {}_{2}Y^{*}_{l'm'}(\Omega)  \right]
$$

$$
\Gamma(p,q) =  \frac{1}{4 \pi} \mathcal{R} \left[  \sum_{l,l'} \sum_{m, m'} A_l A_{l'} \cdot Y^*_{lm}(p) Y_{l'm'}(q) \cdot  \delta_{ll'} \delta_{mm'} \right]
$$

$$
\Gamma(p,q) =  \frac{1}{4 \pi} \mathcal{R} \left[  \sum_{l} \sum_{m} (A_l)^2 \cdot Y^*_{lm}(p) Y_{lm}(q) \right]
$$

$$
\Gamma(p,q) =  \frac{1}{4 \pi} \mathcal{R} \left[  \sum_{l} (A_l)^2 \cdot \frac{2l+1}{4 \pi} \cdot P_l(p \cdot q)\right]
$$

$$
\Gamma(p,q) = \sum_{l} \frac{2l+1}{(l+2)(l+1)l(l-1)} \cdot P_l(p \cdot q)
$$

## 2 Модифицированная кривая

Как можно заметить, основная загвоздка с любым неизотропным источником $\mathcal{P}(\Omega)$ состоит в том, что нам придется интегрировать уже три функции от $\Omega$, поэтому перед этим сделаем одно важное преобразование:

### 2.1 Коэффициенты Клебша-Гордона

Представим произведение двух сферических гармоник как линейную комбинацию от третьей (правило сложения моментов):

$$
{}_{2}Y_{lm} (\Omega) \cdot {}_{2}Y_{l'm'} (\Omega) = \sum_{L = |l -l'|}^{l+l'} \sum_{M = -L}^{L} \sqrt{\frac{(2l+1)(2l'+1)}{4\pi(2L+1)}} <l, 2,l',-2|L,0> <l,m,l',m'|L,M> Y_{LM}(\Omega)
$$

$$
{}_{2}Y_{lm} (\Omega) \cdot {}_{2}Y^{*}_{l'm'} (\Omega) = \sum_{L = |l -l'|}^{l+l'} \sum_{M = -L}^{L} \sqrt{\frac{(2l+1)(2l'+1)}{4\pi(2L+1)}} <l, 2,l',-2|L,0> <l,m,l',-m'|L,M> (-1)^{m'} Y_{LM}(\Omega)
$$

В итоге, интегральное выражение представляется как:

$$
\Gamma(p,q) =  \frac{1}{4 \pi} \mathcal{R} \left[  \sum_{l,l'} \sum_{m, m'} \sum_{L, M} A_l A_{l'} \cdot Y^*_{lm}(p) Y_{l'm'}(q) \cdot  \sqrt{\frac{(2l+1)(2l'+1)}{4\pi(2L+1)}} <l, 2,l',-2|L,0> <l,m,l', -m'|L,M> (-1)^{m'} \int_{S^2} d\Omega \cdot  Y_{LM}(\Omega) \cdot \mathcal{P}(\Omega) \right]
$$

### 2.2 Биполярная сферическая функция

По определению:

$$
\mathcal{Y}^{ll'}_{LM}(p,q) = \sum_{m,m'} <l,m,l', -m'|L,M> Y_{lm}(p) \cdot Y_{l'm'}(q)
$$

Также введем переобозначение:

$$
\mathcal{B}^{ll'}_L = A_l A_{l'} \sqrt{\frac{(2l+1)(2l'+1)}{4\pi(2L+1)}}<l, 2,l',-2|L,0>
$$

Тогда интеграл переписывается как:

$$
\Gamma(p,q) =  \frac{1}{4 \pi} \mathcal{R} \left[  \sum_{l,l'} \sum_{L, M} \mathcal{B}^{ll'}_L \cdot \left(\mathcal{Y}^{ll'}_{LM}(p,q)\right)^{*} \cdot \int_{S^2} d\Omega \cdot  Y_{LM}(\Omega) \cdot \mathcal{P}(\Omega) \right]
$$

Более того, теперь можно поменять порядок суммирования и получить следующее выражение:

$$
\Gamma(p,q) =  \frac{1}{4 \pi} \mathcal{R} \left[ \sum_{L=0}^{\infty} \sum_{M=-L}^{L} \left[ \sum_{l = 2}^{\infty} \sum_{l' = \max(2, |L-l|)}^{L+l} \mathcal{B}^{ll'}_L \cdot \left(\mathcal{Y}^{ll'}_{LM}(p,q)\right)^{*} \right] \cdot \int_{S^2} d\Omega \cdot  Y_{LM}(\Omega) \cdot \mathcal{P}(\Omega) \right]
$$

Переобозначим внутреннюю сумму и интеграл:

$$
\Gamma_{LM}(p,q) =  \frac{1}{4 \pi}  \sum_{l = 2}^{\infty} \sum_{l' = \max(2, |L-l|)}^{L+l} \mathcal{B}^{ll'}_L \cdot \left(\mathcal{Y}^{ll'}_{LM}(p,q)\right)^{*}
$$

$$
\mathcal{P}_{LM}= \int_{S^2} d\Omega \cdot  Y_{LM}(\Omega) \cdot \mathcal{P}(\Omega)
$$

Итоговое выражение:

$$
\Gamma(p,q) = \mathcal{R} \left[ \sum_{L=0}^{\infty} \sum_{M=-L}^{L} \Gamma_{LM}(p,q) \cdot \mathcal{P}_{LM} \right]
$$

## 3 Модельные источники

Рассмотрим значение $\mathcal{P}_{LM}$ для разных вариантов $\mathcal{P}(\Omega)$

### 3.1 Изотропный фон

Пусть $\mathcal{P}(\Omega) = 1$. Тогда:

$$
\mathcal{P}_{LM} = \int_{S^2} d\Omega \cdot  Y_{LM}(\Omega) = \int_{S^2} d\Omega \cdot  Y_{LM}(\Omega) \cdot (\sqrt{4 \pi} \cdot Y_{00}(\Omega))
$$

$$
\mathcal{P}_{LM} = \sqrt{4 \pi}  \int_{S^2} d\Omega \cdot  Y_{LM}(\Omega) \cdot Y_{lm}(\Omega) \cdot \delta_{l0} \delta_{m0} = \sqrt{4 \pi}  \delta_{Ll} \delta_{Mm} \delta_{l0} \delta_{m0} = \sqrt{4 \pi} \delta_{L0} \delta_{M0}
$$

Итого функция $\Gamma(p,q)$ равна:

$$
\Gamma(p,q) =  \frac{1}{4 \pi} \mathcal{R} \left[ \sum_{L=0}^{\infty} \sum_{M=-L}^{L} \Gamma_{LM}(p,q) \cdot \sqrt{4 \pi} \cdot \delta_{L0} \delta_{M0} \right] = \frac{1}{\sqrt{4 \pi}} \mathcal{R} \left[\Gamma_{00}(p,q)\right]
$$

### 3.2 Точечный источник

Пусть $\mathcal{P}(\Omega) = A \cdot \delta(\Omega - \Omega_0)$. Тогда:

$$
\mathcal{P}_{LM} = \int_{S^2} d\Omega \cdot  Y_{LM}(\Omega) \cdot A \cdot \delta(\Omega - \Omega_0) = A \cdot Y_{LM}(\Omega_0)
$$

Итого функция $\Gamma(p,q)$ равна:

$$
\Gamma(p,q) = \frac{1}{4 \pi} \mathcal{R} \left[ \sum_{L=0}^{\infty} \sum_{M=-L}^{L} \Gamma_{LM}(p,q) \cdot A \cdot Y_{LM}(\Omega_0) \right] = \frac{A}{4 \pi} \mathcal{R} \left[ \sum_{L=0}^{\infty} \sum_{M=-L}^{L} \Gamma_{LM}(p,q) \cdot Y_{LM}(\Omega_0) \right]
$$

Но при этом есть аналитическое представление:

$$
\Gamma(p,q) = \frac{A}{4 \pi}\mathcal{R} \left[F_{\hat{p}}(\Omega_0) \cdot F_{\hat{q}}^*(\Omega_0) \right]
$$

### 3.3 Сферическая гармоника

Пусть $\mathcal{P}(\Omega) = p_{\lambda \mu} \cdot Y^*_{\lambda \mu}(\Omega)$.

$$
\mathcal{P}_{LM} = \int_{S^2} d\Omega \cdot  Y_{LM}(\Omega)  \cdot Y^*_{\lambda \mu}(\Omega) \cdot p_{\lambda \mu} = p_{\lambda \mu} \delta_{L \lambda} \delta_{M \mu}
$$

Итого функция $\Gamma(p,q)$ равна:

$$
\Gamma(p,q) = \frac{1}{4 \pi} \mathcal{R} \left[ \sum_{L=0}^{\infty} \sum_{M=-L}^{L} \Gamma_{LM}(p,q) \cdot p_{\lambda \mu} \delta_{L \lambda} \delta_{M \mu}\right] = \frac{p_{\lambda \mu}}{4 \pi} \mathcal{R} \left[\Gamma_{\lambda \mu}(p,q)\right]
$$

### 3.4 Гауссиана

Пусть $\mathcal{P}(\Omega) = G \cdot  \exp \left[ - \frac{(\Omega - \Omega_0)^2}{2 \sigma^2} \right]$. Преобразуем выражение и используем замену $\kappa = 1/\sigma^2$:

$$
(\Omega - \Omega_0)^2 = (\Omega - \Omega_0) \cdot (\Omega - \Omega_0) = \Omega \cdot \Omega + \Omega_0 \cdot \Omega_0 - 2\Omega \cdot \Omega_0  = 1 + 1 - 2\Omega \cdot \Omega_0 = 2 \cdot (1 - \Omega \cdot \Omega_0)
$$

Тогда $\mathcal{P}(\Omega)$ перепишется как:

$$
\mathcal{P}(\Omega) =  G \cdot \exp \left[ - \kappa \cdot\frac{2 \cdot (1 - \Omega \cdot \Omega_0)}{2} \right] =  g\cdot\exp \left[ - \kappa + \kappa \cdot \Omega \cdot \Omega_0) \right] =  g \cdot \exp \left(- \kappa \right) \cdot \exp \left(\kappa \cdot \Omega \cdot \Omega_0 \right)
$$

где:

$$
\exp \left(\kappa \cdot \Omega \cdot \Omega_0 \right) = \sum_{l = 0}^{\infty} b_l P_l(\Omega \cdot \Omega_0 ),
$$

а $b_l(\kappa) = (2l + 1) \cdot i_l(\kappa)$, а $i_l(\kappa)$ - сферическая модифицированная функция Бесселя:

$$
P_l(\Omega \cdot \Omega_0 ) = \frac{4 \pi}{2l + 1} \sum_{m = -l}^l Y_{lm}(\Omega_0) \cdot Y^*_{lm}(\Omega)
$$

Итого:

$$
\mathcal{P}(\Omega) = G \cdot \exp \left(- \kappa \right) \cdot \sum_{l = 0}^{\infty} (2l + 1) \cdot i_l(\kappa) \cdot \frac{4 \pi}{2l + 1} \sum_{m = -l}^l Y_{lm}(\Omega_0) \cdot Y^*_{lm}(\Omega)
$$

$$
\mathcal{P}(\Omega) = \sum_{l = 0}^{\infty} \sum_{m = -l}^l \varepsilon_{lm} \cdot Y^*_{lm}(\Omega)
$$

где коэффициенты равны:

$$
\varepsilon_{lm} = G \cdot g_l(\kappa) \cdot Y_{lm}(\Omega_0)
$$

а функция $g_l(\kappa)$ определяется как:
$$
g_l(\kappa) = 4 \pi \cdot exp(-\kappa) \cdot i_l(\kappa)
$$

Тогда $\mathcal{P}_{LM}$ равно:

$$
\mathcal{P}_{LM} = \sum_{l = 0}^{\infty} \sum_{m = -l}^l \varepsilon_{lm} \int_{S^2} d\Omega \cdot  Y_{LM}(\Omega)  \cdot Y^*_{lm}(\Omega) = \sum_{l = 0}^{\infty} \sum_{m = -l}^l \varepsilon_{lm}  \delta_{L l} \delta_{M m} = \varepsilon_{LM}
$$

$$
\mathcal{P}_{LM} = G \cdot g_L(\kappa) \cdot Y_{LM}(\Omega_0)
$$

Итого функция $\Gamma(p,q)$ равна:

$$
\Gamma(p,q) = \frac{1}{4 \pi} \mathcal{R} \left[ \sum_{L=0}^{\infty} \sum_{M=-L}^{L} \Gamma_{LM}(p,q) \cdot G \cdot g_L(\kappa) \cdot Y_{LM}(\Omega_0)\right]
$$

$$
\Gamma(p,q) = \frac{G}{4 \pi} \mathcal{R} \left[ \sum_{L=0}^{\infty} g_L(\kappa) \sum_{M=-L}^{L} \Gamma_{LM}(p,q) \cdot Y_{LM}(\Omega_0)\right]
$$

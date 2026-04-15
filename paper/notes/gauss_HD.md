# Модифицированная кривая HD для гауссова источника

## 1 Источник

Мощность источника выглядит как:

$$
\mathcal{P}(\Omega) = \exp \left[ - \frac{(\Omega - \Omega_0)^2}{2 \sigma^2} \right]
$$

Распишем числитель:
$$
(\Omega - \Omega_0)^2 = \Omega^2 + \Omega_0^2 - 2 \Omega \cdot \Omega_0 = 1 + 1 -  2 \Omega \cdot \Omega_0 = 2 (1 - \Omega \cdot \Omega_0)
$$

Тогда:

$$
\mathcal{P}(\Omega) = \exp \left[ - \frac{2 (1 - \Omega \cdot \Omega_0)}{2 \sigma^2} \right] = \exp \left[ - \frac{1}{\sigma^2} + \frac{\Omega \cdot \Omega_0 }{\sigma^2}\right] = \exp \left[ - \frac{1}{\sigma^2} \right] \cdot  \exp \left[ \frac{\Omega \cdot \Omega_0 }{\sigma^2}\right]
$$

Введем обозначение $\kappa = 1/\sigma^2$, тогда:

$$
\mathcal{P}(\Omega) = \exp \left(-\kappa \right) \cdot  \exp \left( \kappa \Omega \cdot \Omega_0 \right)
$$

## 2 Представление интеграла

Разложим экспоненту от скалярного произведения в ряд Тейлора (который сходится для любого $x \in \mathbb{R}$). Тогда:

$$
\exp \left( \kappa \Omega \cdot \Omega_0 \right) = \sum_{n=0}^{\infty} \frac{\kappa^n}{n!} \cdot (\Omega \cdot \Omega_0)^n
$$

Интеграл для идеального PTA в бескоординатном (тензорном) виде выглядит так:

$$
\Gamma(p,q) = \beta \int_{S^2} \mathrm{d \Omega} \cdot \mathcal{K}(p,q,\Omega) \cdot \mathcal{P}(\Omega)
$$

Где интегральное ядро $K(p,q,\Omega)$ выглядит как:

$$
\mathcal{K}(p,q,\Omega) = \frac{2 ((p\cdot \Omega)(q \cdot \Omega) - (p \cdot q))^2 - (1 - (p\cdot \Omega)^2)(1 - (q \cdot \Omega)^2)}{4(1 + (p\cdot \Omega))(1 + (q \cdot \Omega))}
$$

$$
\mathcal{K}(p,q,\Omega) = \frac{1}{4} \left[ \frac{2 ((p\cdot \Omega)(q \cdot \Omega) - (p \cdot q))^2}{(1 + (p\cdot \Omega))(1 + (q \cdot \Omega))} -  (1 - (p\cdot \Omega))(1 - (q \cdot \Omega)) \right]
$$

С учетом разложения экспоненты, итоговый интеграл будет выглядеть как:

$$
\Gamma(p,q) = - \frac{\beta}{4} \int_{S^2} \mathrm{d \Omega} \cdot \exp \left( \kappa \Omega \cdot \Omega_0 \right) \cdot  \left[(1 - (p\cdot \Omega))(1 - (q \cdot \Omega)) -  \frac{2 ((p\cdot \Omega)(q \cdot \Omega) - (p \cdot q))^2}{(1 + (p\cdot \Omega))(1 + (q \cdot \Omega))} \right]
$$

$$
\Gamma(p,q) = - \frac{\beta}{4} \sum_{n=0}^{\infty} \frac{\kappa^n}{n!} \cdot  \int_{S^2} \mathrm{d \Omega} \cdot  (\Omega \cdot \Omega_0)^n \cdot  \left[(1 - (p\cdot \Omega))(1 - (q \cdot \Omega)) -  \frac{2 ((p\cdot \Omega)(q \cdot \Omega) - (p \cdot q))^2}{(1 + (p\cdot \Omega))(1 + (q \cdot \Omega))} \right]
$$

Разобъем интеграл внутри суммы на линейную и нелинейную часть:

$$
\Gamma(p,q) = - \frac{\beta}{4} \sum_{n=0}^{\infty} \frac{\kappa^n}{n!} \cdot (I^{(n)} - 2 \cdot J^{(n)})
$$

где:

$$
I^{(n)} =  \int_{S^2} \mathrm{d \Omega} \cdot  (\Omega \cdot \Omega_0)^n \cdot (1 - (p\cdot \Omega)) \cdot (1 - (q \cdot \Omega))
$$

$$
J^{(n)} =  \int_{S^2} \mathrm{d \Omega} \cdot  (\Omega \cdot \Omega_0)^n \cdot \frac{((p\cdot \Omega)(q \cdot \Omega) - (p \cdot q))^2}{(1 + (p\cdot \Omega))(1 + (q \cdot \Omega))}
$$

Эти интегралы нам и необходимо вычислить в конкретной системе координат.

## 3 Система координат

Поскольку в оба интеграла у нас входит скалярное произведение $(\Omega \cdot \Omega_0)^n$, чтобы под степенью был одночлен, нам необходимо закрепить вектор $\Omega_0$ в зените, тем самым выбрав направление оси z. Тогда:

$$
\Omega_0 = (0, 0, 1)
$$

$$
\Omega = (\sin \theta \cos \phi, \sin \theta \sin \phi, \cos \theta)
$$

$$
(\Omega \cdot \Omega_0)^n = \cos^n \theta
$$

Также, нам нужно задать положения двух пульсаров p и q. Поскольку у нас осталась симметрия интеграла относительно поворота вокруг оси z ($\phi \rightarrow \phi + \lambda$), мы можем закрепить вектор p на нулевой долготе. Тогда:

$$
p = (\sin \alpha, 0, \cos \alpha)
$$

$$
q = (\sin \beta \cos \gamma, \sin \beta \sin \gamma, \cos \beta)
$$

где $\alpha$, $\beta$ - полярные углы пульсаров, $\gamma$ - разница долгот пульсаров.

Итого, скалярные произведения будут выглядеть как:

$$
(p \cdot \Omega) = \sin \theta \sin \alpha \cos \phi + \cos \theta \cos \alpha
$$

$$
(q \cdot \Omega) = \sin \theta \sin \beta \cos \gamma \cos \phi + \sin \theta \sin \beta \sin \gamma \sin \phi + \cos \theta \cos \beta
$$

$$
(p \cdot q) = \sin \alpha \sin \beta \cos \gamma + \cos \alpha \cos \beta
$$

## 4 Интеграл $I^{(n)}$ в координатах

Рассмотрим линейный интеграл $I^{(n)}$:

$$
I^{(n)} =  \int_{0}^{\pi} \mathrm{d \theta} \cdot \sin \theta \int_{0}^{2 \pi} \mathrm{d \phi} \cdot (\Omega \cdot \Omega_0)^n \cdot ((p\cdot \Omega) - 1) \cdot ((q \cdot \Omega) - 1)
$$

Поскольку $(\Omega \cdot \Omega_0)^n = \cos^n \theta$, удобнее считать сначала именно интеграл по $\phi$, а потом уже интеграл по $\theta$. А если же мы считаем сначала интеграл по $\phi$, значит для его расчете нам нужно считать $\theta$ как константу, и тогда удобно ввести следующие обозначения:

$$
(p \cdot \Omega) = a \cos \phi + b
$$

$$
(q \cdot \Omega) = f \cos \phi + g \sin \phi + d
$$

$$
(p \cdot q) = h
$$

Где коэффициенты соответственно равны:

$$a = \sin \theta \sin \alpha$$

$$b = \cos \theta \cos \alpha$$

$$c = \sin \theta \sin \beta$$

$$f = \sin \theta \sin \beta \cos \gamma = c \cos \gamma$$

$$g = \sin \theta \sin \beta \sin \gamma =  c \sin \gamma$$

$$d = \cos \theta \cos \beta$$

$$h = \sin \alpha \sin \beta \cos \gamma + \cos \alpha \cos \beta$$

Тогда подынтегральное выражение по $\mathrm{d \phi}$ перепишется как:

$$
I^{(n)}_{\phi} = \int_{0}^{2 \pi} \mathrm{d \phi} \cdot (a \cos \phi + (b - 1))\cdot ( f \cos \phi + g \sin \phi + (d - 1))
$$

$$
I^{(n)}_{\phi} = \int_{0}^{2 \pi} \mathrm{d \phi} \cdot (af \cdot \cos^2 \phi + ag \cdot \cos \phi \sin \phi + [a(d-1) + f(b - 1)] \cos \phi + g(b-1) \sin \phi + (b - 1)(d - 1))
$$

$$
I^{(n)}_{\phi} = 2 \pi \left[ \frac{af}{2} + (b-1)(d-1)\right]
$$

$$
I^{(n)}_{\phi} = \pi \left[ \sin^2 \theta \sin \alpha \sin \beta \cos \gamma + 2 \cos^2 \theta \cos \alpha \cos \beta - 2 \cos \theta \cos \alpha - 2 \cos \theta \cos \beta + 2 \right]
$$

$$
I^{(n)}_{\phi} = \pi \left[\cos^2 \theta \cdot (2  \cos \alpha \cos \beta - \sin \alpha \sin \beta \cos \gamma) - 2 \cos \theta \cdot (\cos \alpha + \cos \beta ) + (2 + \sin \alpha \sin \beta \cos \gamma) \right]
$$

Тогда интеграл $I^{(n)}$ равен:

$$
I^{(n)} = \int_{0}^{\pi} \mathrm{d \theta} \cdot \sin \theta \cdot \cos^n \theta \cdot I^{(n)}_{\phi}(\theta) = - \int_{+1}^{-1} \mathrm{d (\cos \theta)} \cdot \cos^n \theta \cdot I^{(n)}_{\phi}(\theta) = \int_{-1}^{+1} \mathrm{d (\cos \theta)} \cdot \cos^n \theta \cdot I^{(n)}_{\phi}(\theta)
$$

$$
I^{(n)} =  \pi \int_{-1}^{+1} \mathrm{d (\cos \theta)} \cdot \cos^n \theta \cdot  \left[\cos^2 \theta \cdot (2  \cos \alpha \cos \beta - \sin \alpha \sin \beta \cos \gamma) - 2 \cos \theta \cdot (\cos \alpha + \cos \beta ) + (2 + \sin \alpha \sin \beta \cos \gamma) \right]
$$

Пусть $x = \cos \theta$, тогда:

$$
I^{(n)} =  \pi \int_{-1}^{+1} \mathrm{d} x \cdot x^n \cdot  \left[x^2 \cdot (2  \cos \alpha \cos \beta - \sin \alpha \sin \beta \cos \gamma) - 2 x \cdot (\cos \alpha + \cos \beta ) + (2 + \sin \alpha \sin \beta \cos \gamma) \right]
$$

$$
I^{(n)} =  \pi \int_{-1}^{+1} \mathrm{d} x \cdot  \left[x^{n+2} \cdot (2  \cos \alpha \cos \beta - \sin \alpha \sin \beta \cos \gamma) - 2 x^{n+1} \cdot (\cos \alpha + \cos \beta ) + x^n \cdot (2 + \sin \alpha \sin \beta \cos \gamma) \right]
$$

$$
I^{(n)} =  \pi  \left[\frac{x^{n+3}}{n+3} \cdot (2  \cos \alpha \cos \beta - \sin \alpha \sin \beta \cos \gamma) - 2 \frac{x^{n+2}}{n+2} \cdot (\cos \alpha + \cos \beta ) + \frac{x^{n+1}}{n+1} \cdot (2 + \sin \alpha \sin \beta \cos \gamma) \right] \bigg|_{-1}^{+1}
$$

$$
I^{(n)} =  \pi  \left[\frac{(+1)^{n+3} - (-1)^{n+3}}{n+3} \cdot (2  \cos \alpha \cos \beta - \sin \alpha \sin \beta \cos \gamma) - 2 \frac{(+1)^{n+2} - (-1)^{n+2}}{n+2} \cdot (\cos \alpha + \cos \beta ) + \frac{(+1)^{n+1} - (-1)^{n+1}}{n+1} \cdot (2 + \sin \alpha \sin \beta \cos \gamma) \right]
$$

$$
I^{(n)} =  \pi  \left[\frac{1 + (-1)^{n}}{n+3} \cdot (2  \cos \alpha \cos \beta - \sin \alpha \sin \beta \cos \gamma) - 2 \frac{1 + (-1)^{n+1}}{n+2} \cdot (\cos \alpha + \cos \beta ) + \frac{1 + (-1)^{n}}{n+1} \cdot (2 + \sin \alpha \sin \beta \cos \gamma) \right]
$$

В итоге:

$$
I^{(n)} =  4\pi  \left[ \frac{1 + (-1)^{n}}{2} \left( \frac{1}{n+1} (1 + \frac{1}{2} \sin \alpha \sin \beta \cos \gamma) +  \frac{1}{n+3} (\cos \alpha \cos \beta - \frac{1}{2} \sin \alpha \sin \beta \cos \gamma)\right) + \frac{1 + (-1)^{n+1}}{2} \left(\frac{1}{n+2} (\cos \alpha + \cos \beta)\right) \right]
$$

Что характерно, при $n = 0$ и $\sin \alpha = 0$ мы возвращаемся к классической кривой HD и $I^{(0)} = I$, где:

$$
I = 4 \pi \left[1 + \frac{1}{3}\cos \beta \right]
$$

## 5 Интеграл $J^{(n)}$ в координатах

Рассмотрим нелинейный интеграл $J^{(n)}$:

$$
J^{(n)} = \int_{0}^{\pi} \mathrm{d \theta} \cdot \sin \theta \int_{0}^{2 \pi} \mathrm{d \phi} \cdot  (\Omega \cdot \Omega_0)^n \cdot \frac{((p\cdot \Omega)(q \cdot \Omega) - (p \cdot q))^2}{(1 + (p\cdot \Omega))(1 + (q \cdot \Omega))}
$$

Аналогично линейному интегралу, проще сначала брать интеграл по $\mathrm{d \phi}$ и также слагаемое $(\Omega \cdot \Omega_0)^n$ выносится за интеграл по $\phi$.

Распишем числитель $P(\phi)$ и знаменатель $Q(\phi)$ в тех же обозначениях для зависимости от $\phi$:

$$
P(\phi) = (p\cdot \Omega)(q \cdot \Omega) - (p \cdot q) = (a \cos \phi + b)(f \cos \phi + g \sin \phi + d) - h
$$

$$
P(\phi) = af \cdot \cos^2 \phi + ag \cdot \cos \phi \sin \phi + (ad + bf) \cdot \cos \phi + bg \cdot \sin \phi + (bd - h)
$$

$$
Q(\phi) = (p\cdot \Omega + 1)(q \cdot \Omega + 1) = (a \cos \phi + (b + 1))(f \cos \phi + g \sin \phi + (d + 1))
$$

$$
Q(\phi) = af \cdot \cos^2 \phi + ag \cdot \cos \phi \sin \phi + (a(d+1) + f(b+1)) \cdot \cos \phi + g (b + 1) \cdot \sin \phi + (b+1)(d+1)
$$

Заметим, что квадратичная часть по $\cos \phi$  у числителя и знаменателя одинаковая. Более того, мы можем выделить из знаменателя некоторый остаток $R(\phi)$ линейный по синусу и косинусу:

$$
Q(\phi) = af \cdot \cos^2 \phi + ag \cdot \cos \phi \sin \phi + (ad + bf) \cdot \cos \phi + bg \cdot \sin \phi + (bd - h) + (a + f) \cdot \cos \phi + g \cdot \sin \phi + (b + d + 1 + h)
$$

$$
Q(\phi) = P(\phi) + R(\phi)
$$

где $R(\phi)$ равен:

$$
R(\phi) = (a + f) \cdot \cos \phi + g \cdot \sin \phi + (1 + b + d + h)
$$

Тогда интеграл по $\phi$ будет выглядеть как:

$$
J^{(n)}_{\phi} = \int_{0}^{2 \pi} \mathrm{d \phi} \cdot \frac{P(\phi)^2}{Q(\phi)} = \int_{0}^{2 \pi} \mathrm{d \phi} \cdot \frac{(Q(\phi) - R(\phi))^2}{Q(\phi)} = \int_{0}^{2 \pi} \mathrm{d \phi} \cdot \frac{Q(\phi)^2 - 2 Q(\phi) R(\phi) + R(\phi)^2}{Q(\phi)}
$$

$$
J^{(n)}_{\phi} =  \int_{0}^{2 \pi} \mathrm{d \phi} \cdot \left[ Q(\phi) - 2 R(\phi)\right] + \frac{R(\phi)^2}{Q(\phi)} = \int_{0}^{2 \pi} \mathrm{d \phi} \cdot \left[ P(\phi) - R(\phi)\right] + \frac{R(\phi)^2}{Q(\phi)}
$$

Посчитаем теперь $R(\phi)^2$:

$$
R(\phi)^2 = (a^2 + f^2 + 2af) \cdot \cos^2 \phi + g^2 \cdot \sin^2 \phi + (1 + b + d + h)^2  + 2 g (a + f) \cdot \cos \phi \sin \phi + 2 (a + f)(1 + b + d + h) \cdot \cos \phi + 2g(1 + b + d + h) \cdot \sin \phi
$$

$$
R(\phi)^2 = (a^2 + f^2 - g^2 + 2af) \cdot \cos^2 \phi + ((1 + b + d + h)^2 + g^2) + 2 g (a + f) \cdot \cos \phi \sin \phi + 2 (a + f)(1 + b + d + h) \cdot \cos \phi + 2g(1 + b + d + h) \cdot \sin \phi
$$

Из этого выражения снова можно выделить многочлен $P(\phi)$, что уже не изменит размерность уравнения, но немного упростит вычисления:

$$
R(\phi)^2 = 2 \left[ af \cdot \cos^2 \phi + ag \cdot \cos \phi \sin \phi + (ad + bf)\cdot \cos \phi + bg \cdot \sin \phi + bd - h \right] + (a^2 + f^2 - g^2) \cdot \cos^2 \phi + 2 gf \cdot \cos \phi \sin \phi +  2(a(1 + h + b) + f(1 + h + d))\cdot \cos \phi + 2g(1 + h + d) \cdot \sin \phi + (1 + b^2 + d^2 + h^2 + g^2 + 2b + 2d + 2bh + 2dh + 4h) 
$$

$$
R(\phi)^2 = 2 P(\phi) + S(\phi)
$$

где $S(\phi)$ равен:

$$
S(\phi) = (a^2 + f^2 - g^2) \cdot \cos^2 \phi + 2 gf \cdot \cos \phi \sin \phi +  2(a(1 + h + b) + f(1 + h + d))\cdot \cos \phi + 2g(1 + h + d) \cdot \sin \phi + (1 + b^2 + d^2 + h^2 + g^2 + 2b + 2d + 2bh + 2dh + 4h)
$$

Тогда:

$$
\frac{R(\phi)^2}{Q(\phi)} = \frac{2 P(\phi) + S(\phi)}{Q(\phi)} = \frac{2 Q(\phi) - 2R(\phi) + S(\phi)}{Q(\phi)} = 2 + \frac{S(\phi) - 2R(\phi)}{Q(\phi)}
$$

Посчитаем теперь $H(\phi) = S(\phi) - 2R(\phi)$:

$$
H(\phi) = (a^2 + f^2 - g^2) \cdot \cos^2 \phi + 2 gf \cdot \cos \phi \sin \phi +  2(a(h + b) + f(h + d) + a + f)\cdot \cos \phi + 2g(1 + h + d) \cdot \sin \phi + (1 + b^2 + d^2 + h^2 + g^2 + 2b + 2d + 2bh + 2dh + 4h) -  2 (a + f) \cdot \cos \phi - 2g \cdot \sin \phi -2 (1 + b + d + h)
$$

$$
H(\phi) = (a^2 + f^2 - g^2) \cdot \cos^2 \phi + 2 gf \cdot \cos \phi \sin \phi + 2(a(h + b) + f(h + d))\cdot \cos \phi + 2g(h + d) \cdot \sin \phi + (b^2 + d^2 + h^2 + g^2 - 1  + 2h (b + d + 1))
$$

Итого, интеграл $J^{(n)}_{\phi}$ равен:

$$
J^{(n)}_{\phi} = \int_{0}^{2 \pi} \mathrm{d \phi} \cdot \left[ P(\phi) - R(\phi) + 2 \right] + \frac{H(\phi)}{Q(\phi)} = 2\pi \left[\frac{af}{2} + (bd - h) - (1 + b + d + h) + 2 \right] + \int_{0}^{2 \pi} \mathrm{d \phi} \cdot \frac{H(\phi)}{Q(\phi)}
$$

$$
J^{(n)}_{\phi} = 2\pi \left[\frac{af}{2} + bd - b - d - 2h + 1 \right] + \int_{0}^{2 \pi} \mathrm{d \phi} \cdot \frac{H(\phi)}{Q(\phi)}
$$

Итого, мы свели изначальный интеграл по $\phi$ к вычислению интеграла:

$$
K = \int_{0}^{2 \pi} \mathrm{d \phi} \cdot \frac{H(\phi)}{Q(\phi)}
$$

## 6 Интеграл K в комплексной плоскости

Введем естественную замену:

$$
z = e^{i \phi}
$$

Тогда возникают следующие соотношения:

$$
\mathrm{d \phi} = \frac{\mathrm{d z}}{iz}
$$

$$
\cos \phi = \frac{1}{2} \left(z + \frac{1}{z} \right)
$$

$$
\sin \phi = \frac{1}{2i} \left(z - \frac{1}{z} \right)
$$

$$
\cos^2 \phi = \frac{1}{4} \left(z^2 + \frac{1}{z^2} \right) + \frac{1}{2}
$$

$$
\cos \phi \sin \phi = \frac{1}{4i} \left(z^2 - \frac{1}{z^2} \right)
$$

Тогда числитель и знаменатель выглядят как:

$$
\tilde{H}(z) = \frac{(a^2 + f^2 - g^2)}{4} \left(z^2 + \frac{1}{z^2}\right) - \frac{2gfi}{4} \left(z^2 - \frac{1}{z^2} \right) + \frac{(a^2 + f^2 - g^2)}{2} + (a(h + b) + f(h + d)) \left(z + \frac{1}{z} \right) - gi(h + d) \left(z - \frac{1}{z} \right) + (b^2 + d^2 + h^2 + g^2 - 1  + 2h (b + d + 1))
$$

$$
\tilde{H}(z) = \frac{(a^2 + f^2 - g^2 - 2gfi)}{4} \cdot z^2 + \frac{(a^2 + f^2 - g^2 + 2gfi)}{4} \cdot \frac{1}{z^2} + (a(h + b) + (f - gi)(h + d)) \cdot z + (a(h + b) + (f + gi)(h + d))  \cdot \frac{1}{z} + (b^2 + d^2 + h^2 - 1  + 2h (b + d + 1)) + \frac{(a^2 + f^2 + g^2)}{2}
$$

$$
\tilde{H}(z) = \frac{(a^2 + (f - gi)^2)}{4} \cdot z^2 + \frac{(a^2 + (f + gi)^2)}{4} \cdot \frac{1}{z^2} + (a(h + b) + (f - gi)(h + d)) \cdot z + (a(h + b) + (f + gi)(h + d))  \cdot \frac{1}{z} + (b^2 + d^2 + h^2 - 1  + 2h (b + d + 1)) + \frac{(a^2 + f^2 + g^2)}{2}
$$

$$
\tilde{Q}(z) = \frac{af}{4} \left(z^2 + \frac{1}{z^2}\right) - \frac{agi}{4} \left(z^2 - \frac{1}{z^2} \right) + \frac{af}{2} + \frac{(a(d+1) + f(b+1))}{2} \left(z + \frac{1}{z} \right) - \frac{gi(b+1)}{2} \left(z - \frac{1}{z} \right) + (b+1)(d+1)
$$

$$
\tilde{Q}(z) = \frac{a(f - ig)}{4} \cdot z^2 + \frac{a(f + ig)}{4} \cdot \frac{1}{z^2} + \frac{(a(d+1) + (f - ig)(b+1))}{2} \cdot z + \frac{(a(d+1) + (f + ig)(b+1))}{2} \cdot \frac{1}{z} + (b+1)(d+1) + \frac{af}{2}
$$

Также заметим, что комбинация $f - ig$ можно расписать как:

$$
f - ig = c \cos \gamma - i c \sin \gamma = c e^{-i \gamma}
$$

Тогда можно переписать:

$$
\tilde{H}(z) = \frac{(a^2 + c^2 \cdot e^{-i 2\gamma})}{4} \cdot z^2 + \frac{(a^2 + c^2 \cdot e^{+i 2\gamma})}{4} \cdot \frac{1}{z^2} + (a(h + b) + c(h + d)e^{-i \gamma}) \cdot z + (a(h + b) + c(h + d)e^{+i \gamma})  \cdot \frac{1}{z} + (b^2 + d^2 + h^2 - 1  + 2h (b + d + 1)) + \frac{(a^2 + c^2)}{2}
$$

$$
\tilde{Q}(z) = \frac{ac e^{-i \gamma}}{4} \cdot z^2 + \frac{ac e^{+i \gamma}}{4} \cdot \frac{1}{z^2} + \frac{(a(d+1) + c(b+1)e^{-i \gamma})}{2} \cdot z + \frac{(a(d+1) + c(b+1)e^{+i \gamma})}{2} \cdot \frac{1}{z} + (b+1)(d+1) + \frac{af}{2}
$$

$$
\tilde{Q}(z) = \frac{ac}{4}  \cdot \left(\frac{z^2}{e^{i\gamma}} + \frac{e^{i \gamma}}{z^2} \right) + \frac{c(b+1)}{2} \cdot \left(\frac{z}{e^{i \gamma}} + \frac{e^{i \gamma}}{z} \right) + \frac{a(d+1)}{2} \cdot \left(\frac{z}{1} + \frac{1}{z} \right) + (b+1)(d+1) + \frac{ac}{4} \left( \frac{1}{e^{i \gamma}} + \frac{e^{i \gamma}}{1} \right)
$$

Попробуем представить $\tilde{Q}(z)$ в виде произведения множителей:
$$
\tilde{Q}(z) = \frac{ac}{4}  \cdot \left(\frac{z^2}{e^{i\gamma}} + \frac{e^{i \gamma}}{z^2} + \frac{1}{e^{i \gamma}} + \frac{e^{i \gamma}}{1} \right) + \frac{c(b+1)}{2} \cdot \left(\frac{z}{e^{i \gamma}} + \frac{e^{i \gamma}}{z} \right) + \frac{a(d+1)}{2} \cdot \left(\frac{z}{1} + \frac{1}{z} \right) + (b+1)(d+1)
$$

$$
\tilde{Q}(z) = \frac{ac}{4}  \cdot \left(\frac{1}{e^{i\gamma}} \cdot (z^2 + 1) + \frac{e^{i \gamma}}{1} \cdot (\frac{1}{z^2} +1 ) \right) + \frac{c(b+1)}{2} \cdot \left(\frac{z}{e^{i \gamma}} + \frac{e^{i \gamma}}{z} \right) + \frac{a(d+1)}{2} \cdot \left(\frac{z}{1} + \frac{1}{z} \right) + (b+1)(d+1)
$$

$$
\tilde{Q}(z) = \frac{ac}{4}  \cdot \left(\frac{z}{e^{i\gamma}} \cdot \frac{z^2 + 1}{z} + \frac{e^{i \gamma}}{z} \cdot \frac{z^2 + 1}{z}\right) + \frac{c(b+1)}{2} \cdot \left(\frac{z}{e^{i \gamma}} + \frac{e^{i \gamma}}{z} \right) + \frac{a(d+1)}{2} \cdot \left(\frac{z}{1} + \frac{1}{z} \right) + (b+1)(d+1)
$$

$$
\tilde{Q}(z) = \frac{ac}{4} \cdot  \left(\frac{z}{1} + \frac{1}{z} \right) \cdot \left(\frac{z}{e^{i\gamma}}  + \frac{e^{i \gamma}}{z}\right) + \frac{c(b+1)}{2} \cdot \left(\frac{z}{e^{i \gamma}} + \frac{e^{i \gamma}}{z} \right) + \frac{a(d+1)}{2} \cdot \left(\frac{z}{1} + \frac{1}{z} \right) + (b+1)(d+1)
$$

$$
\tilde{Q}(z) = \frac{c}{2} \cdot \left(\frac{z}{e^{i\gamma}}  + \frac{e^{i \gamma}}{z}\right)  \cdot \left[ \frac{a}{2} \cdot \left(\frac{z}{1} + \frac{1}{z} \right) +  (b+1) \right] + (d + 1) \cdot \left[ \frac{a}{2} \cdot \left(\frac{z}{1} + \frac{1}{z} \right) +  (b+1) \right]
$$

$$
\tilde{Q}(z) = \left[\frac{c}{2} \cdot \left(\frac{z}{e^{i\gamma}}  + \frac{e^{i \gamma}}{z}\right) + (d + 1) \right] \cdot \left[ \frac{a}{2} \cdot \left(\frac{z}{1} + \frac{1}{z} \right) +  (b+1) \right]
$$

Каждую скобку теперь тоже можно представить в виде произведения множителей как решение квадратного уравнения:

$$
z + \frac{1}{z} = - \frac{2(b+1)}{a}
$$

$$
z^2 + 2z \cdot \frac{(b+1)}{a} + 1 = 0
$$

$$
D = \frac{(b+1)^2}{a^2} - 1 = \frac{b^2 + 2b + 1 - a^2}{a^2} = \frac{\cos^2 \theta \cos^2 \alpha + 2 \cos \theta \cos \alpha + 1 - \sin^2 \theta \sin^2 \alpha}{\sin^2 \theta \sin^2 \alpha} = \frac{\cos^2 \theta \cos^2 \alpha + \cos^2 \theta \sin^2 \alpha - \sin^2 \alpha + 1 + 2 \cos \theta \cos \alpha }{\sin^2 \theta \sin^2 \alpha}
$$

$$
D = \frac{\cos^2 \theta + \cos^2 \theta + \cos^2 \alpha + 2 \cos \theta \cos \alpha }{\sin^2 \theta \sin^2 \alpha} = \left( \frac{\cos \theta + \cos \alpha}{\sin \theta \sin \alpha} \right)^2
$$

$$
z^{(1)}_{\pm} = \frac{b + 1}{a} \pm \sqrt{D} = \frac{\cos \theta \cos \alpha + 1 \pm (\cos \theta + \cos \alpha)}{\sin \theta \sin \alpha}
$$

$$
z^{(1)}_{+} = \frac{\cos \theta \cos \alpha + 1 + \cos \theta + \cos \alpha}{\sin \theta \sin \alpha} = \frac{(\cos \theta + 1)(\cos \alpha + 1)}{\sin \theta \sin \alpha}
$$

$$
z^{(1)}_{-} = \frac{\cos \theta \cos \alpha + 1 - \cos \theta -  \cos \alpha}{\sin \theta \sin \alpha} = \frac{(\cos \theta - 1)(\cos \alpha - 1)}{\sin \theta \sin \alpha}
$$

Для второй скобки решение аналогичное, с точностью до замены $z \rightarrow z/e^{i \gamma}$ и $\alpha \rightarrow \beta$:

$$
z^{(2)}_{+} = \frac{(\cos \theta + 1)(\cos \beta + 1)}{\sin \theta \sin \beta} \cdot e^{i \gamma}
$$

$$
z^{(2)}_{-} = \frac{(\cos \theta - 1)(\cos \beta - 1)}{\sin \theta \sin \beta} \cdot e^{i \gamma}
$$

Введем также вспомогательные многочлены для корректной работы со степенями:

$$
\hat{H}(z) = \tilde{H}(z) \cdot z^2
$$

$$
\hat{Q}(z) = \tilde{Q}(z) \cdot z^2
$$

Тогда наш интеграл K будет равен:

$$
K =\int_{0}^{2 \pi} \mathrm{d \phi} \cdot \frac{H(\phi)}{Q(\phi)} = \oint_{|z| = 1} \frac{\mathrm{dz}}{iz} \cdot \frac{\tilde{H}(z)}{\tilde{Q}(z)} = \oint_{|z| = 1} \frac{\mathrm{dz}}{i} \cdot \frac{\hat{H}(z)}{z\hat{Q}(z)}
$$

Поскольку нам известны нули (полюса) знаменателя, мы можем сказать что интеграл по контуру будет равен вычетам в этих точках, если корни лежат внутри единичной окружности:

$$
K = 2 \pi i \cdot \frac{1}{i} \sum_{|z_k| < 1} \mathord{Res}\left(\frac{\hat{H}(z)}{z\hat{Q}(z)}, z_k \right)
$$

где $z_k = {0, z^{(1)}_{+}, z^{(1)}_{-}, z^{(2)}_{+}, z^{(2)}_{-}}$. Поскольку все эти полюса первого порядка, то работает следующая формула:

$$
\mathord{Res}\left(\frac{f(z)}{g(z)}, z_k \right) = \frac{f(z_k)}{g'(z_k)}
$$

где соответственно $g(z_k) = 0$ и $f(z_k) \neq 0$. Тогда для функции $g(z) = z \cdot \hat{Q}(z)$ полюса будут считаться как:

$$
\mathord{Res} \left( \frac{\hat{H}(z)}{z\hat{Q}(z)}, z_k  \right) = \frac{H(z_k)}{z_k \cdot \hat{Q}'(z_k) + \hat{Q}(z_k)}
$$

Итого, интеграл будет равен:

$$
K = 2 \pi \left[\frac{\hat{H}(0)}{\hat{Q(0)}} + \sum_{|z^{\pm}_{k}| < 1} \frac{H(z_k)}{z_k \cdot \hat{Q}'(z_k)} \right]
$$

Определение нужного корня происходит следующим образом:

$$
z_1 = \frac{(1 \mp \cos \theta)(1 \mp \cos \alpha)}{\sin \theta \sin \alpha}
$$

где "-" соответствует диапазону $0 < \theta < \pi - \alpha$, а "+" соответствует диапазону $\pi - \alpha < \theta < \pi$

Аналогично для $z_2$:

$$
z_2 = \frac{(1 \mp \cos \theta)(1 \mp \cos \beta)}{\sin \theta \sin \beta} \cdot e^{i \gamma}
$$

где "-" соответствует диапазону $0 < \theta < \pi - \beta$, а "+" соответствует диапазону $\pi - \beta < \theta < \pi$ 

Тогда итого интеграл выглядит как:

$$
K = 2 \pi \left[ \frac{\hat{H}(0)}{\hat{Q(0)}} + \frac{\hat{H}(z_1)}{z_1 \cdot \hat{Q}'(z_1)} + \frac{\hat{H}(z_2)}{z_2 \cdot \hat{Q}'(z_2)} \right]
$$


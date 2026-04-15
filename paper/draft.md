# The Hellings-Downs curve for a Gaussian source

PDF Version is avaliable here:
[https://raw.githubusercontent.com/arikallis-j/pta-sim/main/paper/paper.pdf](https://raw.githubusercontent.com/arikallis-j/pta-sim/main/paper/paper.pdf)

This paper is devoted to the derivation of the Hellings-Downs curve for the case of a Gaussian source of the Gravitational Wave Background.

## 1. Problem Statement

Consider the spherical coordinate system $(\phi, \theta, r)$, where $\phi$ is longitude, $\theta$ is colatitude and r is radius. Then the orthonormal basis in the cartesian coordinate system will look like:

$$\begin{align}
    \hat{\Omega} &= \left(\sin \theta \cos \phi, \sin \theta \sin \phi, \cos \theta \right), \\
    \hat{n} &= (\cos \theta \cos \phi, \cos \theta \sin \phi, - \sin \theta), \\
    \hat{m} &= (\sin \phi, - \cos \phi, 0),
\end{align}$$

where $(\hat{\Omega}, \hat{n}, \hat{m}) \equiv (\hat{r}, \hat{\theta}, -\hat{\phi})$. Since $\bar{r} = r \cdot \hat{r}$, and the point on the unit sphere has the value $r = 1$, we can describe the unit sphere with a single vector $\hat{\Omega} = \hat{r}$.

We also know that gravitational waves have two polarization, which are called "plus" and "cross", and are described by the corresponding tensors:

$$\begin{align}
    e^+_{\alpha \beta}(\hat{\Omega}) &= \hat{m}_{\alpha} \hat{m}_{\beta} - \hat{n}_{\alpha} \hat{n}_{\beta}, \\
    e^{\times}_{\alpha \beta}(\hat{\Omega}) &= \hat{m}_{\alpha} \hat{n}_{\beta} + \hat{n}_{\alpha} \hat{m}_{\beta}
\end{align}$$

Then, frequency shift of pulsar radiation $z = \Delta \nu/\nu_0$ in Fourier space:

$$\begin{equation}
    \tilde{z}(f, \hat{\Omega}) = \left( e^{- i 2 \pi f L (1 + \hat{\Omega} \cdot \hat{p})} - 1\right) h(f, \hat{\Omega}) F_{\hat{p}}(f, \hat{\Omega}),
\end{equation}$$

where $\hat{p}$ is the unit vector in the direction of the pulsar, L is the distance to the pulsar, $h(f, \hat{\Omega})$ is spectrum of gravitational waves and $F_{\hat{p}}(\hat{\Omega})$ is pulsar response:

$$\begin{equation}
F_{\hat{p}}(\hat{\Omega}) = e_{\alpha \beta}(\hat{\Omega}) \frac{1}{2} \frac{\hat{p}^{\alpha} \hat{p}^{\beta}}{1 + \hat{\Omega} \cdot \hat{p}}.
\end{equation}$$

Here we use a complex representation of the polarization tensor and spectrum, and in the case where the GWB is unpolarized, we have:

$$\begin{align}
    h(f, \hat{\Omega}) &= h_{+}(f, \hat{\Omega}) + i \cdot h_{\times}(f, \hat{\Omega}) \\
    e_{\alpha \beta}(\hat{\Omega}) &= e^{+}_{\alpha \beta}(\hat{\Omega}) + i \cdot e^{\times}_{\alpha \beta}(\hat{\Omega})
\end{align}$$

Since the Hellings-Downs curve relates correlations between signals from two pulsars, we defined the Hellings-Downs curve as:

$$\begin{equation}
\Gamma[\mathcal{P}](\hat{p}, \hat{q}) = \frac{3}{4\pi}\int_{S^2} d \hat{\Omega} \cdot \mathcal{K}(\hat{\Omega}, \hat{p}, \hat{q}) \cdot \mathcal{P}(\hat{\Omega}),
\end{equation}$$

where $\mathcal{P}(\hat{\Omega})$ is the power spectrum on the unit sphere, and with the frequency power spectrum $\mathcal{H}(f)$ defined as:

$$\begin{equation}
|h(f, \hat{\Omega})|^2  = \mathcal{H}(f) \cdot \mathcal{P}(\hat{\Omega}),
\end{equation}$$

and $\mathcal{K}(\hat{\Omega},\hat{p}, \hat{q})$ is the integral kernel, which is defined as:

$$\begin{equation}
\mathcal{K}(\hat{\Omega},\hat{p}, \hat{q}) = \mathcal{R}\left[F^{*}_{\hat{p}}(\hat{\Omega}) \cdot F_{\hat{q}}(\hat{\Omega})\right].
\end{equation}$$

And in our case, we assume that the power spectrum is Gaussian and looks like this:

$$\begin{equation}
\mathcal{P}(\hat{\Omega}) = \exp \left[ - \frac{(\hat{\Omega} - \hat{\Omega}_0)^2}{2 \sigma^2} \right],
\end{equation}$$

where $\hat{\Omega}_0$ is the center of the source and $\sigma$ is the width of the Gaussian. Converting the numerator:

$$
(\hat{\Omega} - \hat{\Omega}_0)^2 = \hat{\Omega}^2 + \hat{\Omega}_0^2 - 2 \hat{\Omega} \cdot \hat{\Omega}_0 = 1 + 1 -  2 \hat{\Omega} \cdot \hat{\Omega}_0
$$

$$
(\hat{\Omega} - \hat{\Omega}_0)^2 = 2 (1 - \hat{\Omega} \cdot \hat{\Omega}_0),
$$

we have:

$$
\mathcal{P}(\hat{\Omega}) = \exp \left[ - \frac{2 (1 - \hat{\Omega} \cdot \hat{\Omega}_0)}{2 \sigma^2} \right],
$$

$$
\mathcal{P}(\hat{\Omega}) = \exp \left[ - \frac{1}{\sigma^2} \right] \cdot  \exp \left[ \frac{\hat{\Omega} \cdot \hat{\Omega}_0 }{\sigma^2}\right].
$$

Let $\kappa = 1/\sigma^2$, and as a result we have:

$$\begin{equation}
\mathcal{P}(\hat{\Omega}) = \exp \left(-\kappa \right) \cdot  \exp \left( \kappa \hat{\Omega} \cdot \hat{\Omega}_0 \right).
\end{equation}$$

And to find the Hellings-Downs curve for a Gaussian source, we need to calculate:

$$\begin{equation}
\Gamma^{(g)}(\hat{p}, \hat{q}) = \frac{3 e^{-\kappa}}{4\pi}\int_{S^2} d \hat{\Omega} \cdot \mathcal{K}(\hat{\Omega}, \hat{p}, \hat{q}) \cdot e^{\kappa \hat{\Omega} \cdot \hat{\Omega}_0}
\end{equation}$$

## Integral in Coordinates

Firstly, let's represent the integral kernel $\mathcal{K}$ in vector form:

$$\begin{equation*}
\mathcal{K}= \mathcal{R}\left[ e^{*}_{\alpha \beta}(\hat{\Omega}) \frac{1}{2} \frac{\hat{p}^{\alpha} \hat{p}^{\beta}}{1 + \hat{\Omega} \cdot \hat{p}} \cdot e_{\mu \nu}(\hat{\Omega}) \frac{1}{2} \frac{\hat{q}^{\mu} \hat{q}^{\nu}}{1 + \hat{\Omega} \cdot \hat{q}}
\right],
\end{equation*}$$

$$\begin{equation*}
\mathcal{K}= \frac{1}{4} \frac{\hat{p}^{\alpha} \hat{p}^{\beta}\hat{q}^{\mu} \hat{q}^{\nu}}{(1 + \hat{\Omega} \cdot \hat{p})(1 + \hat{\Omega} \cdot \hat{q})} \mathcal{R}\left[e^{*}_{\alpha \beta}(\hat{\Omega}) \cdot  e_{\mu \nu}(\hat{\Omega})\right],
\end{equation*}$$

where:

$$\begin{equation*}
\mathcal{R}\left[e^{*}_{\alpha \beta}(\hat{\Omega}) \cdot  e_{\mu \nu}(\hat{\Omega})\right] = e^{+}_{\alpha \beta}(\hat{\Omega})e^{+}_{\mu \nu}(\hat{\Omega}) + e^{\times}_{\alpha \beta}(\hat{\Omega})e^{\times}_{\mu \nu}(\hat{\Omega}),
\end{equation*}$$

then:

$$\begin{equation}
\mathcal{K} = \frac{1}{4} \frac{\hat{p}^{\alpha} \hat{p}^{\beta}\hat{q}^{\mu} \hat{q}^{\nu} \left[ e^{+}_{\alpha \beta}(\hat{\Omega})e^{+}_{\mu \nu}(\hat{\Omega}) + e^{\times}_{\alpha \beta}(\hat{\Omega})e^{\times}_{\mu \nu}(\hat{\Omega}) \right]}{(1 + \hat{\Omega} \cdot \hat{p})(1 + \hat{\Omega} \cdot \hat{q})}.
\end{equation}$$

Consider individual tensor convolutions:

$$\begin{align}
\hat{v}^{\alpha} \hat{v}^{\beta} e^{+}_{\alpha \beta}(\hat{\Omega}) &= (\hat{v} \cdot \hat{m})^2 - (\hat{v} \cdot \hat{n})^2, \\
\hat{v}^{\alpha} \hat{v}^{\beta} e^{\times}_{\alpha \beta}(\hat{\Omega}) &= 2 (\hat{v} \cdot \hat{m})(\hat{v} \cdot \hat{n}).
\end{align}$$

Let's denote $P_{a} = (\hat{p} \cdot \hat{a})$ and $Q_{a} = (\hat{q} \cdot \hat{a})$, then:

$$\begin{equation}
\mathcal{K} = \frac{1}{4} \frac{(P_m^2 - P_n^2)\cdot(Q_m^2 - Q_n^2) + 4 P_m P_n Q_m Q_n}{(1 + P_{\Omega})(1 + Q_{\Omega})}.
\end{equation}$$

Note that since $(\hat{m}, \hat{n}, \hat{\Omega})$ is an orthonormal basis, and $|\hat{p}| = 1$, $|\hat{q}| = 1$, it means that:

$$\begin{align*}
    P_n^2 &= 1 - P_m^2 - P_{\Omega}^2, \\
    Q_n^2 &= 1 - Q_m^2 - Q_{\Omega}^2.
\end{align*}$$

Then:

$$\begin{align*}
\frac{(P_n^2 - P_m^2)}{1 + P_{\Omega}} &= \frac{1 - 2 P_m^2 - P_{\Omega}^2}{1 + P_{\Omega}} = (1 - P_{\Omega}) - \frac{2 P_m^2}{1 + P_{\Omega}}, \\
\frac{(Q_n^2 - Q_m^2)}{1 + Q_{\Omega}} &= \frac{1 - 2 Q_m^2 - Q_{\Omega}^2}{1 + Q_{\Omega}} = (1 - Q_{\Omega}) - \frac{2Q_m^2}{1 + Q_{\Omega}}.
\end{align*}$$

As a result, we can divide the integral kernel into four parts:

$$\begin{equation}
\mathcal{K}(\hat{\Omega},\hat{p}, \hat{q}) = \frac{1}{4}(\mathcal{K}_{I} + \mathcal{K}_{J} + \mathcal{K}_{K} + \mathcal{K}_{H}),
\end{equation}$$

where:

$$\begin{align}
\mathcal{K}_{I} &= (1 - P_{\Omega})(1 - Q_{\Omega}), \\
\mathcal{K}_{J} &=  - 2 \cdot \frac{P_m^2 (1 - Q_{\Omega})}{(1+P_{\Omega})}, \\
\mathcal{K}_{K} &=  - 2 \cdot \frac{Q_m^2 (1 - P_{\Omega})}{(1+Q_{\Omega})}, \\
\mathcal{K}_{H} &= 4 \cdot \frac{P_m Q_m (P_m Q_m + P_n Q_n)}{(1+P_{\Omega})(1+Q_{\Omega})},
\end{align}$$

and we can divide the integral into the same four parts:

$$\begin{equation}
\Gamma^{(g)}(\hat{p}, \hat{q}) = \frac{3 e^{-\kappa}}{4\pi} \cdot \frac{1}{4} \left( I + J + K + H \right),
\end{equation}$$

where:

$$\begin{align}
I &= \int_{S^2} d\hat{\Omega} \cdot \mathcal{K}_{I} \cdot e^{\kappa \hat{\Omega} \cdot \hat{\Omega}_0} \\
J &= \int_{S^2} d\hat{\Omega} \cdot \mathcal{K}_{J} \cdot e^{\kappa \hat{\Omega} \cdot \hat{\Omega}_0} \\
K &= \int_{S^2} d\hat{\Omega} \cdot \mathcal{K}_{K} \cdot e^{\kappa \hat{\Omega} \cdot \hat{\Omega}_0} \\
H &= \int_{S^2} d\hat{\Omega} \cdot \mathcal{K}_{H} \cdot e^{\kappa \hat{\Omega} \cdot \hat{\Omega}_0}
\end{align}$$

Let's define a convenient coordinate system. As we can see, all integrals include a term containing $(\hat{\Omega} \cdot \hat{\Omega}_0)$ in the exponent, and to simplify this term, we take $\hat{\Omega}_0 \equiv (0, 0, 1)$, because then $(\hat{\Omega} \cdot \hat{\Omega}_0) = \cos \theta$. For the pulsar vectors, we have:

$$\begin{align}
    \hat{p} &= (\sin \alpha, 0, \cos \alpha) \\
    \hat{q} &= (\sin \beta \cos \gamma, \sin \beta \sin \gamma, \cos \beta)
\end{align}$$

where $\alpha$, $\beta$ are the angles between the source center and the pulsars, and $\gamma$ is the angle between the directions to the pulsars from the source center. Now we can calculate all the scalar products:

$$\begin{align*}
    P_{\Omega} &= \sin \alpha \sin \theta \cos \phi + \cos \alpha \cos \theta, \\
    Q_{\Omega} &= \sin \beta \sin \theta \cos (\phi - \gamma) + \cos \beta \cos \theta, \\
    P_{n} &= \sin \alpha \cos \theta \cos \phi - \cos \alpha \sin \theta, \\
    Q_{n} &= \sin \beta \cos \theta \cos (\phi - \gamma) - \cos \beta \sin \theta \\
    P_{m} &= \sin \alpha \sin \phi, \\
    Q_{m} &= \sin \beta \sin (\phi - \gamma);
\end{align*}$$

Also, let's highlight the dependence of the coefficients on $\phi$:

$$\begin{align}
    P_{\Omega} &= a \cos \phi + b, Q_{\Omega} = c \cos (\phi - \gamma) + d, \\
    P_{n} &= h \cos \phi - s, Q_{n} = k \cos (\phi - \gamma) - t, \\
    P_{m} &= f \sin \phi, Q_{m} = g \sin (\phi - \gamma);
\end{align}$$

where:

$$\begin{align}
    a &= \sin \alpha \sin \theta, b = \cos \alpha \cos \theta, \\
    c &= \sin \beta \sin \theta, d = \cos \beta \cos \theta, \\
    h &= \sin \alpha \cos \theta, s = \cos \alpha \sin \theta, \\
    k &= \sin \alpha \cos \theta, t = \cos \alpha \sin \theta, \\
    f &= \sin \alpha , g = \sin \beta,
\end{align}$$

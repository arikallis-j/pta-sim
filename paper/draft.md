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

## 2. Integral in Coordinates

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

and define scalar product between $\hat{p}$ and $\hat{q}$ as:
$$\begin{equation}
    \cos \xi = \sin \alpha \sin \beta \cos \gamma + \cos \alpha \cos \beta,
\end{equation}$$

where $\xi$ is the angle between $\hat{p}$ and $\hat{q}$.

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
    k &= \sin \beta \cos \theta, t = \cos \beta \sin \theta, \\
    f &= \sin \alpha , g = \sin \beta,
\end{align}$$

## 3. Calculation of Integrals

All integrals are calculated first in longitude and then in latitude, and in all of them, after the first integration, we get an integral only in $\cos\theta$, and the integral can be replaced by an integral from $-1$ to $+1$ in $x$. Let's calculate the integral of I using these steps:

$$\begin{equation*}
I = \int_{S^2} d\hat{\Omega} \cdot \mathcal{K}_{I} \cdot e^{\kappa \hat{\Omega} \cdot \hat{\Omega}_0}
\end{equation*}$$

$$\begin{equation}
I = \int_{0}^{\pi} \sin \theta \cdot d \theta \cdot e^{\kappa \cdot \cos \theta} \int_{0}^{2\pi} d \phi \cdot \mathcal{K}_{I}
\end{equation}$$

$$\begin{equation}
I_{\phi} = \int_{0}^{2\pi} d \phi \cdot \mathcal{K}_{I}
\end{equation}$$

The result for $I_{\phi}$ is:

$$\begin{equation}
I_{\phi} = 2\pi \cdot \left[ (b-1)(d-1) + \frac{1}{2}\cdot ac \cos \gamma\right]
\end{equation}$$

and after highlighting the dependency on $\cos\theta$:

$$\begin{equation}\begin{aligned}
I_{\phi} = 2\pi \cdot \Bigl[\cos^2 \theta \cdot \left(\frac{3}{2}\cos \alpha \cos \beta - \frac{1}{2} \cos \xi \right) \\ - \cos \theta \cdot \left(\cos \alpha + \cos \beta \right)\\ + \left(1 + \frac{1}{2} \cos \xi  - \frac{1}{2} \cos \alpha \cos \beta \right) \Bigr]
\end{aligned}\end{equation}$$

Then we can rewrite the final integral as an integral in $x$:

$$\begin{equation}\begin{aligned}
I = \int_{0}^{\pi} (-1) \cdot d(\cos \theta) \cdot e^{\kappa \cdot \cos \theta} \cdot I_{\phi}(\cos \theta) \\ = \int_{-1}^{1} dx \cdot I_{\phi}(x) \cdot e^{\kappa x}.
\end{aligned}\end{equation}$$

As a result, the integral $I$ will be equal to:

$$\begin{equation}\begin{aligned}
I = 4 \pi \Bigl[ \frac{\sinh\kappa}{\kappa} + \frac{3(\kappa \cosh \kappa - \sinh \kappa)}{\kappa^3} \cdot \frac{\cos \xi}{3} \\ + \left(\frac{\sinh \kappa }{\kappa} - \frac{3(\kappa \cosh \kappa - \sinh \kappa)}{\kappa^3}\right) \cdot \cos \alpha \cos \beta  \\ - \frac{\kappa \cosh \kappa - \sinh \kappa}{\kappa^2} \cdot (\cos \alpha + \cos \beta) \Bigr]
\end{aligned}\end{equation}$$

Let's calculate the integral $J$ using the same algorithm:

$$\begin{equation*}
J = \int_{S^2} d\hat{\Omega} \cdot \mathcal{K}_{J} \cdot e^{\kappa \hat{\Omega} \cdot \hat{\Omega}_0}
\end{equation*}$$

$$\begin{equation}
J = \int_{0}^{\pi} \sin \theta \cdot d \theta \cdot e^{\kappa \cdot \cos \theta} \int_{0}^{2\pi} d \phi \cdot \mathcal{K}_{J}
\end{equation}$$

$$\begin{equation}
J_{\phi} = \int_{0}^{2\pi} d \phi \cdot \mathcal{K}_{J}
\end{equation}$$

The result for $J_{\phi}$ is:

$$\begin{equation}\begin{aligned}
J_{\phi} = 4\pi f^2 \cdot \Bigl[ \frac{(b+1) - \sqrt{(b+1)^2 - a^2}}{a^2} \cdot (d-1) \\ + \left(\frac{(b+1) - \sqrt{(b+1)^2 - a^2}}{a^2}\right)^2 \cdot \frac{ac \cos \gamma}{2} \Bigr]
\end{aligned}\end{equation}$$

and after highlighting the dependency on $\cos\theta$:

$$\begin{equation}\begin{aligned}
J_{\phi} = 4\pi \cdot \Bigl[ \frac{1 + \cos \alpha \cos \theta - |\cos \alpha + \cos \theta|}{1 - \cos^2 \theta} \cdot (\cos \beta \cos \theta - 1) \\ + \frac{\left(1 + \cos \alpha \cos \theta - |\cos \alpha + \cos \theta|\right)^2}{1 - \cos^2 \theta} \cdot \frac{\cos \xi - \cos \alpha \cos \beta}{2 (1 - \cos \alpha^2)}\Bigr]
\end{aligned}\end{equation}$$

Then we can rewrite the final integral as an integral in $x$:

$$\begin{equation}\begin{aligned}
J = \int_{0}^{\pi} (-1) \cdot d(\cos \theta) \cdot e^{\kappa \cdot \cos \theta} \cdot J_{\phi}(\cos \theta) \\ = \int_{-1}^{1} dx \cdot J_{\phi}(x) \cdot e^{\kappa x}
\end{aligned}\end{equation}$$

As a result, the integral $J$ will be equal to:

$$\begin{equation}\begin{aligned}
J = 4 \pi \Bigl[ \frac{(1-\cos \alpha) \cdot (1 + \cos \alpha + \cos \beta + \cos \xi)}{1 + \cos \alpha} \cdot \\  e^{-\kappa} \cdot \left(\text{Ei} [\kappa(1 - \cos \alpha)] - \text{Ei}[2\kappa] \right) \\ + \frac{(1+\cos \alpha) \cdot (1 - \cos \alpha - \cos \beta + \cos \xi)}{1 - \cos \alpha} \cdot \\ e^{\kappa} \cdot \left(\text{Ei} [- \kappa(1 + \cos \alpha)] - \text{Ei}[-2\kappa] \right) \\ + (1-\cos \alpha) \cdot \left(\cos \beta + \frac{1}{2}(\cos \alpha \cos \beta + \cos \xi)\right) \cdot \\ \left( \frac{\sinh \kappa}{\kappa} + \cos \alpha \cdot \frac{\cosh \kappa - \exp(-\kappa \cdot \cos \alpha)}{\kappa \cdot \cos \alpha} \right) \cdot \frac{1}{1 + \cos \alpha} \\ - (1+\cos \alpha) \cdot \left(\cos \beta - \frac{1}{2}(\cos \alpha \cos \beta + \cos \xi)\right) \cdot \\ \left( \frac{\sinh \kappa}{\kappa} - \cos \alpha \cdot \frac{\cosh \kappa - \exp(-\kappa \cdot \cos \alpha)}{\kappa \cdot \cos \alpha} \right) \cdot \frac{1}{1 - \cos \alpha} \Bigr]
\end{aligned}\end{equation}$$

And since the integral $K$ has the same structure, we can write the final equation for $K$ by simply replacing $\cos\alpha$ and $\cos\beta$ in the equation for the integral $J$:

$$\begin{equation}\begin{aligned}
K = 4 \pi \Bigl[ \frac{(1-\cos \beta) \cdot (1 + \cos \alpha + \cos \beta + \cos \xi)}{1 + \cos \beta} \cdot \\ e^{-\kappa} \cdot \left(\text{Ei} [\kappa(1 - \cos \beta)] - \text{Ei}[2\kappa] \right) \\ + \frac{(1+\cos \beta) \cdot (1 - \cos \alpha - \cos \beta + \cos \xi)}{1 - \cos \beta} \cdot \\ e^{\kappa} \cdot \left(\text{Ei} [- \kappa(1 + \cos \beta)] - \text{Ei}[-2\kappa] \right) \\ + (1-\cos \beta) \cdot \left(\cos \alpha + \frac{1}{2}(\cos \alpha \cos \beta + \cos \xi)\right) \cdot \\ \left( \frac{\sinh \kappa}{\kappa} + \cos \beta \cdot \frac{\cosh \kappa - \exp(-\kappa \cdot \cos \beta)}{\kappa \cdot \cos \beta} \right) \cdot \frac{1}{1 + \cos \beta}  \\ - (1+\cos \beta) \cdot \left(\cos \alpha - \frac{1}{2}(\cos \alpha \cos \beta + \cos \xi)\right) \cdot \\ \left( \frac{\sinh \kappa}{\kappa} - \cos \beta \cdot \frac{\cosh \kappa - \exp(-\kappa \cdot \cos \beta)}{\kappa \cdot \cos \beta} \right) \cdot \frac{1}{1 - \cos \beta} \Bigr]
\end{aligned}\end{equation}$$

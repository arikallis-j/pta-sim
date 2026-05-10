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
\mathcal{P}(\hat{\Omega}) \sim \exp \left[ - \frac{(\hat{\Omega} - \hat{\Omega}_0)^2}{2 \sigma^2} \right],
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
\mathcal{P}(\hat{\Omega}) \sim \exp \left[ - \frac{2 (1 - \hat{\Omega} \cdot \hat{\Omega}_0)}{2 \sigma^2} \right],
$$

$$
\mathcal{P}(\hat{\Omega}) \sim \exp \left[ - \frac{1}{\sigma^2} \right] \cdot  \exp \left[ \frac{\hat{\Omega} \cdot \hat{\Omega}_0 }{\sigma^2}\right].
$$

Let $\kappa = 1/\sigma^2$, and as a result we have:

$$\begin{equation}
\mathcal{P}(\hat{\Omega}) \sim \exp \left(-\kappa \right) \cdot  \exp \left( \kappa \hat{\Omega} \cdot \hat{\Omega}_0 \right).
\end{equation}$$

And let's normalize our spectrum:

$$\begin{equation}
\frac{1}{4 \pi} \cdot \int_{S^2} d \hat{\Omega} \cdot \mathcal{P}(\hat{\Omega}) = 1
\end{equation}$$

where:

$$\begin{equation}
\mathcal{P}(\hat{\Omega}) = G (\kappa) \cdot  \exp \left( \kappa \hat{\Omega} \cdot \hat{\Omega}_0 \right)
\end{equation}$$

And to find the Hellings-Downs curve for a Gaussian source, we need to calculate:

$$\begin{equation}
\Gamma^{(g)}(\hat{p}, \hat{q}) = \frac{3 \cdot G (\kappa)}{4\pi}\int_{S^2} d \hat{\Omega} \cdot \mathcal{K}(\hat{\Omega}, \hat{p}, \hat{q}) \cdot e^{\kappa \hat{\Omega} \cdot \hat{\Omega}_0}
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
\Gamma^{(g)}(\hat{p}, \hat{q}) = \frac{3 \cdot G (\kappa)}{4\pi} \cdot \frac{1}{4} \left( I + J + K + H \right),
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

All integrals are calculated first in longitude and then in latitude, and in all of them, after the first integration, we get an integral only in $\cos\theta$, and the integral can be replaced by an integral from $-1$ to $+1$ in $x$. Firstly, let's find the normalization $G(\kappa)$ of the spectrum:

$$\begin{equation*}
\frac{1}{4 \pi} \cdot \int_{S^2} d \hat{\Omega} \cdot G (\kappa) \cdot  e^{\kappa \hat{\Omega} \cdot \hat{\Omega}_0} = 1
\end{equation*}$$

$$\begin{equation}
\frac{1}{G(\kappa)} = \frac{1}{4 \pi} \int_{0}^{\pi} \sin \theta \cdot d \theta \cdot e^{\kappa \cdot \cos \theta} \int_{0}^{2\pi} d \phi
\end{equation}$$

$$\begin{equation}
\frac{1}{G(\kappa)} = \frac{1}{2} \cdot \int_{0}^{\pi} \sin \theta \cdot d \theta \cdot e^{\kappa \cdot \cos \theta}
\end{equation}$$

Then we can rewrite the final integral as an integral in $x$:

$$\begin{equation}
\frac{1}{G(\kappa)} = \frac{1}{2} \int_{0}^{\pi} (-1) \cdot d(\cos \theta) \cdot e^{\kappa \cdot \cos \theta} = \frac{1}{2} \int_{-1}^{1} dx \cdot e^{\kappa x}
\end{equation}$$

$$\begin{equation}
\frac{1}{G(\kappa)} = \frac{1}{2\kappa} \int_{-\kappa}^{\kappa} dy \cdot e^{y} = \frac{e^{\kappa} - e^{-\kappa}}{2 \kappa} = \frac{\sinh(\kappa)}{\kappa}.
\end{equation}$$

Next, let's calculate the integral of I using these steps:

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
I = 4 \pi \Bigl[\frac{\sinh\kappa}{\kappa} \cdot (1 + \cos \alpha \cos \beta) \\ + \frac{\kappa \cosh \kappa - \sinh \kappa}{\kappa^3} \cdot (\cos \xi - 3 \cos \alpha \cos \beta) \\ - \frac{\kappa \cosh \kappa - \sinh \kappa}{\kappa^2} \cdot (\cos \alpha + \cos \beta) \Bigr]
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
J = 4 \pi \Bigl[ \frac{(1-\cos \alpha) \cdot (1 + \cos \alpha + \cos \beta + \cos \xi)}{1 + \cos \alpha} \cdot \\ e^{-\kappa} \cdot \left(\text{Ei} [\kappa(1 - \cos \alpha)] - \text{Ei}[2\kappa] \right) \\+ \frac{(1+\cos \alpha) \cdot (1 - \cos \alpha - \cos \beta + \cos \xi)}{1 - \cos \alpha} \cdot \\ e^{\kappa} \cdot \left(\text{Ei} [- \kappa(1 + \cos \alpha)] - \text{Ei}[-2\kappa] \right) \\ +  2 \cdot \left(\cos \xi + \frac{\cos \alpha \cos \beta - \cos \xi}{1 - \cos^2 \alpha}\right) \cdot \\ \left(\frac{\cosh \kappa - \exp(-\kappa \cdot \cos \alpha)}{\kappa \cdot \cos \alpha} - \frac{\sinh \kappa}{\kappa} \right) \\ - (\cos \alpha \cos \beta - \cos \xi) \cdot \frac{\sinh \kappa}{\kappa} \Bigr]
\end{aligned}\end{equation}$$

And since the integral $K$ has the same structure, we can write the final equation for $K$ by simply replacing $\cos\alpha$ and $\cos\beta$ in the equation for the integral $J$:

$$\begin{equation}\begin{aligned}
K = 4 \pi \Bigl[ \frac{(1-\cos \beta) \cdot (1 + \cos \alpha + \cos \beta + \cos \xi)}{1 + \cos \beta} \cdot \\ e^{-\kappa} \cdot \left(\text{Ei} [\kappa(1 - \cos \beta)] - \text{Ei}[2\kappa] \right) \\ + \frac{(1+\cos \beta) \cdot (1 - \cos \alpha - \cos \beta + \cos \xi)}{1 - \cos \beta} \cdot \\ e^{\kappa} \cdot \left(\text{Ei} [- \kappa(1 + \cos \beta)] - \text{Ei}[-2\kappa] \right) \\ +  2 \cdot \left(\cos \xi + \frac{\cos \alpha \cos \beta - \cos \xi}{1 - \cos^2 \beta}\right) \cdot \\ \left(\frac{\cosh \kappa - \exp(-\kappa \cdot \cos \beta)}{\kappa \cdot \cos \beta} - \frac{\sinh \kappa}{\kappa} \right) \\ - (\cos \alpha \cos \beta - \cos \xi) \cdot \frac{\sinh \kappa}{\kappa} \Bigr]
\end{aligned}\end{equation}$$

And finally, let's calculate the integral $H$:

$$\begin{equation*}
H = \int_{S^2} d\hat{\Omega} \cdot \mathcal{K}_{H} \cdot e^{\kappa \hat{\Omega} \cdot \hat{\Omega}_0}
\end{equation*}$$

$$\begin{equation}
H = \int_{0}^{\pi} \sin \theta \cdot d \theta \cdot e^{\kappa \cdot \cos \theta} \int_{0}^{2\pi} d \phi \cdot \mathcal{K}_{H}
\end{equation}$$

$$\begin{equation}
H_{\phi} = \int_{0}^{2\pi} d \phi \cdot \mathcal{K}_{H}
\end{equation}$$

The result for $H_{\phi}$ is:

$$\begin{equation}\begin{aligned}
H_{\phi} = 8\pi fg \cdot \Bigl[
  A_H^{(1)} \cdot H_{\phi}^{(1)} + \frac{A_H^{(2)} \cdot H_{\phi}^{(2)} - A_H^{(3)}\cdot H_{\phi}^{(3)}}{B_H}\Bigr]
\end{aligned}\end{equation}$$

where:

$$\begin{equation}\begin{aligned}
A_H^{(1)} = \frac{ac \cos \gamma}{2} \cdot \left[\frac{(b+1)-\sqrt{(b+1)^2 - a^2}}{a^2}\right] \cdot  \\ \left[\frac{(d+1)-\sqrt{(d+1)^2 - c^2}}{c^2}\right]
\end{aligned}\end{equation}$$

$$\begin{equation}\begin{aligned}
A_H^{(2)} = ac\cos\gamma - \left[(b+1)-\sqrt{(b+1)^2 - a^2}\right] \cdot \\ \left[(d+1)-\sqrt{(d+1)^2 - c^2}\right]
\end{aligned}\end{equation}$$

$$\begin{equation}\begin{aligned}
A_H^{(3)} = ac\cos\gamma
\end{aligned}\end{equation}$$

$$\begin{equation}\begin{aligned}
B_H = \sqrt{(b+1)^2 - a^2}\sqrt{(d+1)^2 - c^2} \\ - ac\cos \gamma + (b+1)(d+1)
\end{aligned}\end{equation}$$

$$\begin{equation}\begin{aligned}
H_{\phi}^{(1)} = (hk + fg)\cdot \Bigl(\frac{(b+1)-\sqrt{(b+1)^2 - a^2}}{a^2} \\ \cdot \frac{(d+1)-\sqrt{(d+1)^2 - c^2}}{c^2}\Bigr) \\ - 4 h k \frac{(b+1)(d+1)}{a^2 c^2} \\ - 4 f g \frac{\sqrt{(b+1)^2 - a^2}\sqrt{(d+1)^2 - c^2}}{a^2 c^2}
\end{aligned}\end{equation}$$

$$\begin{equation}\begin{aligned}
H_{\phi}^{(2)} = th \frac{c(b+1)+a(d+1) \cos \gamma}{a^2c^2} \\ + sk \frac{a(d+1)+ c(b+1)\cos \gamma}{a^2c^2} \\ + (hk - fg)\cdot\frac{(b+1)(d+1)}{a^2c^2} + st \frac{1}{ac}
\end{aligned}\end{equation}$$

$$\begin{equation}\begin{aligned}
H_{\phi}^{(3)} = th \frac{c\sqrt{(b+1)^2 - a^2} + a\sqrt{(d+1)^2 - c^2} \cos \gamma}{a^2c^2} \\ + sk \frac{a\sqrt{(d+1)^2 - c^2} + c\sqrt{(b+1)^2 - a^2}\cos\gamma}{a^2c^2} \\ - fg \cdot\frac{\sqrt{(b+1)^2 - a^2}(d+1) + \sqrt{(d+1)^2 - c^2}(b+1)}{a^2c^2}
\end{aligned}\end{equation}$$

and after highlighting the dependency on $\cos\theta$:

$$\begin{equation}\begin{aligned}
H_{\phi} = 8 \pi \cdot (\cos\alpha \cos\beta - \cos\xi) \cdot \\ \Bigl[
\frac{1}{2} - \frac{1}{1 - \cos\theta} \cdot \frac{(1 - \cos\alpha \cos\beta)(\cos\alpha +\cos\beta)}{(1-\cos^2\alpha)(1-\cos^2\beta)} \\ + \frac{C_H^{(1)}}{1-\cos^2\theta} \cdot \frac{1}{1-\cos^2\alpha} + \frac{C_H^{(2)}}{1-\cos^2\theta} \cdot \frac{1}{1-\cos^2\beta} \\ + \frac{1 - \cos\xi}{D_H} \cdot \left(1 + \frac{C_H^{(1)}\cdot C_H^{(2)}}{(1-\cos^2\theta)(\cos\alpha \cos\beta -  \cos\xi)}\right)
\Bigr]
\end{aligned}\end{equation}$$

where:

$$\begin{equation}\begin{aligned}
C_H^{(1)} = |\cos\alpha + \cos\theta| - (\cos\alpha \cos\theta + 1)
\end{aligned}\end{equation}$$

$$\begin{equation}\begin{aligned}
C_H^{(2)} = |\cos\beta + \cos\theta| - (\cos\beta \cos\theta + 1)
\end{aligned}\end{equation}$$

$$\begin{equation}\begin{aligned}
D_H = |\cos\alpha+\cos\theta|\cdot |\cos\beta+ \cos\theta| \\ + (\cos\alpha \cos\theta+1)(\cos\beta \cos\theta +1) \\+ (1-\cos^2\theta)(\cos\alpha \cos\beta - \cos\xi)
\end{aligned}\end{equation}$$

Then we can rewrite the final integral as an integral in $x$:

$$\begin{equation}\begin{aligned}
H = \int_{0}^{\pi} (-1) \cdot d(\cos \theta) \cdot e^{\kappa \cdot \cos \theta} \cdot H_{\phi}(\cos \theta) \\ = \int_{-1}^{1} dx \cdot H_{\phi}(x) \cdot e^{\kappa x}
\end{aligned}\end{equation}$$

As a result, the integral $H$ will be equal to:

$$\begin{equation}\begin{aligned}
H = 4 \pi \Bigl[2\cdot(\cos \alpha \cos \beta - \cos \xi) \cdot \frac{\sinh \kappa}{\kappa} \\ - \frac{(1-\cos \alpha) \cdot (1 + \cos \alpha + \cos \beta + \cos \xi)}{1 + \cos \alpha} \cdot \\  e^{-\kappa} \cdot \left(\text{Ei} [\kappa(1 - \cos \alpha)] - \text{Ei}[2\kappa] \right) \\ - \frac{(1+\cos \alpha) \cdot (1 - \cos \alpha - \cos \beta + \cos \xi)}{1 - \cos \alpha} \cdot \\ e^{\kappa} \cdot \left(\text{Ei} [- \kappa(1 + \cos \alpha)] - \text{Ei}[-2\kappa] \right) \\ - \frac{(1-\cos \beta) \cdot (1 + \cos \alpha + \cos \beta + \cos \xi)}{1 + \cos \beta} \cdot \\ e^{-\kappa} \cdot \left(\text{Ei} [\kappa(1 - \cos \beta)] - \text{Ei}[2\kappa] \right) \\ - \frac{(1+\cos \beta) \cdot (1 - \cos \alpha - \cos \beta + \cos \xi)}{1 - \cos \beta} \cdot \\ e^{\kappa} \cdot \left(\text{Ei} [- \kappa(1 + \cos \beta)] - \text{Ei}[-2\kappa] \right) \\ + (1 - \cos \xi) \cdot \\  2 \cdot \mathcal{R} \left[ e^{-\kappa \cdot r} \cdot \left(\text{Ei} [\kappa(r - \cos \alpha)] + \text{Ei}[\kappa(r - \cos \beta)] \right) \right] \\ - (1 - \cos \xi) \cdot \\ 2 \cdot \mathcal{R} \left[ e^{-\kappa \cdot r} \cdot \left(\text{Ei} [\kappa(r - 1)] + \text{Ei}[\kappa(r + 1)] \right) \right] \Bigr]
\end{aligned}\end{equation}$$

where:

$$\begin{equation}\begin{aligned}
r = \frac{\cos \alpha + \cos \beta + i \cdot \hat{S}}{1 + \cos \xi},
\end{aligned}\end{equation}$$

and

$$\begin{equation}\begin{aligned}
\hat{S} = \sqrt{1 + 2 \cos \alpha \cos \beta \cos \xi - \cos^2 \alpha - \cos^2 \beta - \cos^2 \xi} \\ = |(\hat{p}\times \hat{q})\cdot \hat{\Omega}_0|
\end{aligned}\end{equation}$$

Then the sum of the four integrals is:

$$\begin{equation}\begin{aligned}
I + J + K + H = 4 \pi \Bigl[\frac{\sinh\kappa}{\kappa} \cdot (1 + \cos \alpha \cos \beta) \\ + \frac{\kappa \cosh \kappa - \sinh \kappa}{\kappa^3} \cdot (\cos \xi - 3 \cos \alpha \cos \beta) \\ - \frac{\kappa \cosh \kappa - \sinh \kappa}{\kappa^2} \cdot (\cos \alpha + \cos \beta) \\ +  2 \cdot \left(\cos \xi + \frac{\cos \alpha \cos \beta - \cos \xi}{1 - \cos^2 \alpha}\right) \cdot \\ \left(\frac{\cosh \kappa - \exp(-\kappa \cdot \cos \alpha)}{\kappa \cdot \cos \alpha} - \frac{\sinh \kappa}{\kappa} \right) \\ +  2 \cdot \left(\cos \xi + \frac{\cos \alpha \cos \beta - \cos \xi}{1 - \cos^2 \beta}\right) \cdot \\ \left(\frac{\cosh \kappa - \exp(-\kappa \cdot \cos \beta)}{\kappa \cdot \cos \beta} - \frac{\sinh \kappa}{\kappa} \right) \\ + (1 - \cos \xi) \cdot \\  2 \cdot \mathcal{R} \left[ e^{-\kappa \cdot r} \cdot \left(\text{Ei} [\kappa(r - \cos \alpha)] + \text{Ei}[\kappa(r - \cos \beta)] \right) \right] \\ - (1 - \cos \xi) \cdot \\ 2 \cdot \mathcal{R} \left[ e^{-\kappa \cdot r} \cdot \left(\text{Ei} [\kappa(r - 1)] + \text{Ei}[\kappa(r + 1)] \right) \right]\Bigr]
\end{aligned}\end{equation}$$

And the final integral $\Gamma^{(g)}(\hat{p}, \hat{q})$ will be equal to:

$$\begin{equation}\begin{aligned}
\Gamma^{(g)}(\hat{p}, \hat{q}) = 1 + \frac{3(1 - \cos \xi)}{2} \cdot \left[\mathcal{L}^{(g)}(\hat{p}, \hat{q}) - \frac{1}{6} \right] + \frac{3}{2} \cdot \mathcal{D}^{(g)}(\hat{p}, \hat{q})
\end{aligned}\end{equation}$$

where:

$$\begin{equation}\begin{aligned}
\mathcal{L}^{(g)}(\hat{p}, \hat{q}) = \mathcal{R} \Bigl[ \frac{\kappa \cdot e^{-\kappa \cdot r}}{\sinh \kappa} \cdot \Bigl( \text{Ei} [\kappa(r - \cos \alpha)] + \text{Ei}[\kappa(r - \cos \beta)] \\ -  \text{Ei} [\kappa(r - 1)] - \text{Ei}[\kappa(r + 1)] \Bigr) \Bigr]
\end{aligned}\end{equation}$$

$$\begin{equation}\begin{aligned}
\mathcal{D}^{(g)}(\hat{p}, \hat{q}) = \frac{\cos \beta - \cos \alpha \cos \xi}{1 - \cos^2 \alpha} \cdot \Bigl[\coth \kappa - a \cdot \left(1 + \frac{e^{-\kappa a}}{a \sinh \kappa}\right) \Bigr] \\ + \frac{\cos \alpha - \cos \beta \cos \xi}{1 - \cos^2 \beta} \cdot \Bigl[\coth \kappa - b \cdot \left(1 + \frac{e^{-\kappa b}}{b \sinh \kappa} \right) \Bigr] \\ + \frac{\cos \xi - 3 \cos \alpha \cos \beta}{6} \cdot \Bigl[\frac{3 \coth \kappa}{\kappa} - \frac{3}{\kappa^2} - 1\Bigr] \\ - \frac{\cos \alpha + \cos \beta}{2} \cdot \left[\coth \kappa - \frac{1}{\kappa}\right]
\end{aligned}\end{equation}$$

## A. How to calculate the integrals

For further calculations, we will use the standard integrals, where $k \in \mathbb{Z}$, $n, m \in \mathbb{N}_0$ and $- 1 < a, c < 1$:

$$\begin{equation}\begin{aligned}
V_k(a) = \int_{0}^{2 \pi} \frac{\cos k\phi \cdot d\phi}{a\cdot \cos \phi + 1}
\end{aligned}\end{equation}$$

$$\begin{equation}\begin{aligned}
W_m^n(a) = \int_{0}^{2 \pi} \frac{\sin^n \phi \cdot \cos^m \phi \cdot d \phi }{a\cdot \cos \phi + 1}
\end{aligned}\end{equation}$$

$$\begin{equation}\begin{aligned}
Z(a, c, \gamma) = \int_{0}^{2 \pi} \frac{d \phi}{(a\cdot \cos \phi + 1)(c \cdot \cos (\phi - \gamma) + 1)}
\end{aligned}\end{equation}$$

### A.1. Integral V

We can find the integral $V_k(a)$ using a two-dimensional Poisson kernel:

$$\begin{equation}\begin{aligned}
P_r(\phi) = \sum_{n = -\infty}^{\infty} r^{|n|} e^{in\phi} = \frac{1 - r^2}{1 - 2r\cos \phi + r^2},
\end{aligned}\end{equation}$$

where $0 \le r < 1$. Let's rewrite the denominator in the same way:

$$\begin{equation}\begin{aligned}
1 + a \cdot\cos \phi = A (1 + r^2 - 2r\cos\phi)
\end{aligned}\end{equation}$$

$$\begin{equation*}\begin{aligned}
\begin{cases}
    A \cdot (1 + r^2) = 1 \\
    A \cdot (-2r) = a
\end{cases}
\end{aligned}\end{equation*}$$

$$\begin{equation}\begin{aligned}
    A = \frac{1}{1 + r^2}
\end{aligned}\end{equation}$$

$$\begin{equation*}\begin{aligned}
-\frac{2}{a}r = \frac{1}{A} = 1 + r^2
\end{aligned}\end{equation*}$$

$$\begin{equation*}\begin{aligned}
 r^2 + \frac{2}{a}r + 1 = 0
\end{aligned}\end{equation*}$$

$$\begin{equation}\begin{aligned}
 r = \frac{-1 + \sqrt{1 - a^2}}{a}
\end{aligned}\end{equation}$$

Then we get:

$$\begin{equation*}\begin{aligned}
\frac{1}{1 + a \cdot\cos \phi} = \frac{1 + r^2}{1 - r^2} \cdot \frac{1-r^2}{1 + r^2 - 2r\cos \phi}
\end{aligned}\end{equation*}$$

$$\begin{equation}\begin{aligned}
\frac{1}{1 + a \cdot\cos \phi} = \frac{1 + r^2}{1 - r^2} \sum_{n = -\infty}^{\infty} r^{|n|} e^{in\phi}
\end{aligned}\end{equation}$$

where:

$$\begin{equation*}\begin{aligned}
\frac{1 + r^2}{1 - r^2} = - \frac{2r}{a} \cdot \frac{1}{- \frac{2r}{a} \cdot \sqrt{1 - a^2}} = \frac{1}{\sqrt{1 - a^2}}
\end{aligned}\end{equation*}$$

And let's rewrite the numerator:

$$\begin{equation*}\begin{aligned}
\cos k \phi = \frac{1}{2} \cdot(e^{ik\phi} + e^{-ik\phi})
\end{aligned}\end{equation*}$$

Then the integral $V_k(a)$ will be equal to:

$$\begin{equation*}\begin{aligned}
V_k(a) = \frac{1}{2\sqrt{1 - a^2}} \sum_{n = -\infty}^{\infty} r^{|n|} \int_{0}^{2 \pi} e^{in\phi} \cdot(e^{ik\phi} + e^{-ik\phi}) \cdot d\phi
\end{aligned}\end{equation*}$$

$$\begin{equation*}\begin{aligned}
V_k(a) = \frac{1}{2\sqrt{1 - a^2}} \sum_{n = -\infty}^{\infty} r^{|n|} \int_{0}^{2 \pi} (e^{i\phi(k+n)} + e^{i\phi(n-k)}) \cdot d\phi
\end{aligned}\end{equation*}$$

$$\begin{equation*}\begin{aligned}
V_k(a) = \frac{2\pi}{\sqrt{1 - a^2}} \sum_{n = -\infty}^{\infty} r^{|n|} \cdot \frac{\delta_{n,k} + \delta_{n,-k}}{2}  
\end{aligned}\end{equation*}$$

$$\begin{equation*}\begin{aligned}
V_k(a) = \frac{2\pi}{\sqrt{1 - a^2}} \cdot \frac{r^{|k|} + r^{|-k|}}{2}
\end{aligned}\end{equation*}$$

$$\begin{equation}\begin{aligned}
V_k(a) = \frac{2\pi}{\sqrt{1 - a^2}} \cdot \left(\frac{-1 + \sqrt{1 - a^2}}{a}\right)^{|k|}
\end{aligned}\end{equation}$$

### A.2. Integral W

The first thing we notice is that for $n = 2k+1$ the integral is zero, because:

$$\begin{equation*}\begin{aligned}
W_m^{2k+1}(a) = \int_{0}^{2 \pi} d \phi \cdot \sin \phi \cdot \frac{\sin^{2k} \phi \cdot \cos^m \phi}{a\cdot \cos \phi + 1} \\=  \int_{0}^{2 \pi} d \phi \cdot \sin (-\phi) \cdot \frac{\sin^{2k}(-\phi) \cdot \cos^m (-\phi)}{a\cdot \cos (-\phi) + 1} = - W_m^{2k+1}(a)
\end{aligned}\end{equation*}$$

$$\begin{equation}\begin{aligned}
W_m^{2k+1}(a) = 0
\end{aligned}\end{equation}$$

and also, let's add a formula for reduction:

$$\begin{equation}\begin{aligned}
W_m^{n}(a) = W_m^{n-2}(a) - W_{m+2}^{n-2}(a),
\end{aligned}\end{equation}$$

because:

$$\begin{equation}\begin{aligned}
\sin^n(\phi) \cos^m(\phi) =  \sin^{n-2}(\phi) \cos^m(\phi) - \sin^{n-2}(\phi) \cos^{m+2}(\phi)  
\end{aligned}\end{equation}$$

Next, we can rewrite $\sin^{2k}\phi$  and $\cos^m\phi$ through $\cos k\phi$ like this:

$$\begin{equation*}\begin{aligned}
\sin^{2k}(\phi) = \frac{\left(e^{i\phi} - e^{-i\phi} \right)^{2k}}{(2i)^{2k}} = \frac{(-1)^k}{2^{2k}} \cdot \sum_{s=0}^{2k} \binom{2k}{s} \cdot (e^{i\phi})^{2k-s} (- e^{-i\phi})^{s}
\end{aligned}\end{equation*}$$

$$\begin{equation}\begin{aligned}
\sin^{2k}(\phi) = \frac{(-1)^k}{2^{2k}} \cdot \sum_{s=0}^{2k} \binom{2k}{s} \cdot (-1)^{s} \cdot e^{i\phi (2k - 2s)}
\end{aligned}\end{equation}$$

$$\begin{equation*}\begin{aligned}
\cos^m(\phi) = \frac{1}{2^m} \cdot \left( e^{i\phi} + e^{-i\phi} \right)^m = \frac{1}{2^m} \cdot \sum_{t=0}^m \binom{m}{t} (e^{i\phi})^{m-t} (e^{-i\phi})^{t}
\end{aligned}\end{equation*}$$

$$\begin{equation}\begin{aligned}
\cos^m(\phi) = \frac{1}{2^m} \cdot \sum_{t=0}^m \binom{m}{t} \cdot e^{i\phi(m-2t)}
\end{aligned}\end{equation}$$

As a result, the numerator will look like:

$$\begin{equation}\begin{aligned}
\sin^{2k}(\phi) \cos^m(\phi) =  \frac{(-1)^k}{2^{2k+m}} \cdot \sum_{s=0}^{2k} \sum_{t=0}^m \binom{2k}{s}\binom{m}{t} \cdot (-1)^{s} \cdot \\ e^{i\phi(2k+m - 2s - 2t)}
\end{aligned}\end{equation}$$

And the integral will be equal to:

$$\begin{equation}\begin{aligned}
W_m^{2k}(a) =  \frac{(-1)^k}{2^{2k+m}} \cdot \sum_{s=0}^{2k} \sum_{t=0}^m \binom{2k}{s}\binom{m}{t} \cdot (-1)^{s} \cdot  V_{2k+m - 2s - 2t}(a)
\end{aligned}\end{equation}$$

where $V_{-k}(a) = V_{k}(a)$.

### A.3. Integral Z

This integral is calculated in the complex plane, and the final result looks like this:

$$\begin{equation}\begin{aligned}
Z(a, c, \gamma) = \frac{2\pi}{1 - ac\cos\gamma + \sqrt{1-a^2}\sqrt{1-c^2}} \cdot \\ \left(\frac{1}{\sqrt{1-a^2}} + \frac{1}{\sqrt{1-c^2}} \right)
\end{aligned}\end{equation}$$

Also note that any integral of the form:

$$\begin{equation}\begin{aligned}
Y^n_m(a, c, \gamma) = \int_{0}^{2 \pi} \frac{\sin^n \phi \cdot \cos^m \phi \cdot d \phi}{(a\cdot \cos \phi + 1)(c \cdot \cos (\phi - \gamma) + 1)}
\end{aligned}\end{equation}$$

Can be rewritten as:

$$\begin{equation}\begin{aligned}
Y^n_m(a, c, \gamma) = D\cdot Z(a,c,\gamma) + \sum_{k, s=0}^{N-2}\sum_{s=0}^{N-2} B_k^{s} \cdot W_k^s(0)  \\ + \sum_{k=0}^{N-1}\sum_{s=0}^{N-1} A_k^s \cdot W_k^s(a) + \sum_{k=0}^{N-1}\sum_{s=0}^{N-1} C_k^s \cdot W_k^s(c)
\end{aligned}\end{equation}$$

where $N = \max(n, m)$, because the numerator and denominator in $Y_m^n$ are polynomials in ($\cos \phi$, $\sin \phi$) and we can write:

$$\begin{equation}\begin{aligned}
\frac{P_N(x, y)}{L_1(x,y)\cdot L_2(x,y)} = \frac{P_0}{L_1(x,y)\cdot L_2(x,y)} + P_{N-2}(x,y) \\+ \frac{P_{N-1}^{(1)}(x,y)}{L_1(x,y)} + \frac{P_{N-1}^{(2)}(x,y)}{L_2(x,y)}
\end{aligned}\end{equation}$$

where $x = \cos \phi$, $y = \sin \phi$, and:

$$\begin{equation}\begin{aligned}
\begin{cases}
    P_N(x,y) = x^n y^m \\
    L_1(x,y) = xa + 1 \\
    L_2(x,y) = xc\cos \gamma  + yc\sin\gamma + 1
\end{cases}
\end{aligned}\end{equation}$$

And after reduction for $W_m^n(x)$, we can say that all our integrals are represented as:

$$\begin{equation}\begin{aligned}
K(a, c, \gamma) = D\cdot Z(a,c,\gamma) + \sum_{k}^{N-2} B_k' \cdot W_k^0(0)  \\ + \sum_{k=0}^{N-1} A_k' \cdot W_k^0(a) + \sum_{k=0}^{N-1} C_k' \cdot W_k^0(c)
\end{aligned}\end{equation}$$

Or in terms of $V_k(x)$:

$$\begin{equation}\begin{aligned}
K(a, c, \gamma) = D\cdot Z(a,c,\gamma) + B \cdot V_0(0)  \\ + \sum_{k=0}^{N-1} \left( A_k \cdot V_k(a) + C_k \cdot V_k(c) \right)
\end{aligned}\end{equation}$$

where $V_0(0) = 2 \pi$. As the result, for the integrals $I, J, K, H$ we will have:

$$\begin{equation}\begin{aligned}
I_{\phi}(a, b, c, d, \gamma) = 2\pi \cdot B^{(I)}
\end{aligned}\end{equation}$$

$$\begin{equation}\begin{aligned}
J_{\phi}(a, b, c, d, \gamma) = 2\pi \cdot \frac{B^{(J)}}{b+1}  + \sum_{k=0}^{3} \frac{A_k^{(J)}} {b+1} \cdot V_k\left(\frac{a}{b+1}\right)
\end{aligned}\end{equation}$$

$$\begin{equation}\begin{aligned}
K_{\phi}(a, b, c, d, \gamma) = 2\pi \frac{B^{(K)}}{d+1} + \sum_{k=0}^{3} \frac{C_k^{(K)}} {d+1} \cdot V_k\left(\frac{c}{d+1}\right)
\end{aligned}\end{equation}$$

$$\begin{equation}\begin{aligned}
H_{\phi}(a, b, c, d, \gamma) = 2\pi \cdot \frac{B^{(H)}}{(b+1)(d+1)} \\+ \frac{D^{(H)}}{(b+1)(d+1)}\cdot Z\left(\frac{a}{b+1},\frac{c}{d+1},\gamma\right)  \\+\sum_{k=0}^{3} \frac{A_k^{(H)}} {(b+1)(d+1)} \cdot V_k\left(\frac{a}{b+1}\right) \\ + \sum_{k=0}^{3} \frac{C_k^{(H)}} {(b+1)(d+1)} \cdot V_k\left(\frac{c}{d+1}\right)
\end{aligned}\end{equation}$$

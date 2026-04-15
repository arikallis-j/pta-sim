# The Hellings-Downs curve for a Gaussian source

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
    e^{\times}_{\alpha \beta}(\hat{\Omega}) &= \hat{m}_{\alpha} \hat{n}_{\beta} - \hat{n}_{\alpha} \hat{m}_{\beta}
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
(\hat{\Omega} - \hat{\Omega}_0)^2 = \hat{\Omega}^2 + \hat{\Omega}_0^2 - 2 \hat{\Omega} \cdot \hat{\Omega}_0 = 1 + 1 -  2 \hat{\Omega} \cdot \hat{\Omega}_0 = 2 (1 - \hat{\Omega} \cdot \hat{\Omega}_0),
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

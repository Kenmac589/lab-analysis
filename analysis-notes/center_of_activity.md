# Center of Activity Calculation

This calculation is also derived from the paper by @Martino2014 (Martino, G., Ivanenko, Y. P., et al. (2014) Locomotor Patterns in Cerebellar Ataxia.)

- Below are excerpts from the @Martino2014 paper

> The CoA during the gait cycle was calculated using circular statistics (Batschelet 1981) and plotted in polar coordinates (polar direction denoted the phase of the gait cycle, with angle $\theta$  that varies from 0 to 360°). The CoA of the EMG waveform was calculated as the angle of the vector (1st trigonometric moment) that points to the center of mass of that circular distribution using the following formulas:

$$
\begin{align}
    A &= \sum_{t = 1}^{200} (\cos{\theta_t} \times \text{EMG}_t)
\end{align}
$$
$$
\begin{align}
    B &= \sum_{t = 1}^{200} (\sin{\theta_t} \times \text{EMG}_t)
\end{align}
$$
$$
\begin{align}
    \text{CoA} &= \tan^{-1}(B \mathbin{/} A)
\end{align}
$$

> The CoA was chosen because it was impractical to reliably identify a single peak of activity in the majority of muscles, especially in pathological subjects. It can only be considered as a qualitative parameter, because averaging between distinct foci of activity may lead to misleading activity in the intermediate zone. Nevertheless, it can be helpful to understand if the distribution of muscular activity remains unaltered across different groups and muscles.
